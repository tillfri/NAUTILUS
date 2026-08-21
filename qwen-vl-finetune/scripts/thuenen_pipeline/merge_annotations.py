"""Refine the Thünen ground truth from Megalodon + NAUTILUS box proposals.

Three sources of truth, each trusted for a different thing:

* **The ground truth** decides *which* objects exist and *what they are*. A human
  looked at the frame; nothing here overrules that. No GT box is ever dropped.
* **The proposals** decide *where the box goes*. Their labels are discarded
  entirely -- Megalodon only knows ``object``, and NAUTILUS's zero-shot labelling
  collapses (960 of 1571 test predictions came back ``starfish`` against a true
  support of 262, and ``brittle star``, the most frequent class, got zero). The
  geometry is the part that is worth keeping.
* **Megalodon's surplus boxes** decide which objects the human *missed*. Thünen
  annotations are sparse rather than exhaustive: the extra boxes largely sit on
  real, clearly visible, never-annotated animals.

Why the geometry needs fixing at all: 3386 of 9293 BIIGLE shapes were drawn with
the circle tool, and ``annotations.py:shape_to_box`` turns a circle into its
axis-aligned bounding **square**. 24.9% of GT boxes are exactly square against 2.0%
of NAUTILUS's, and GT boxes run 2.09x the area of a tight box -- so a perfectly
tight prediction inside the median GT box scores IoU = 1/2.09 = 0.478 and is
counted as a false positive *and* a false negative at once.

The merge, per image:

1. Proposals from both runs are converted into one normalised coordinate space and
   stripped of their labels.
2. The two sources are **fused**: where a Megalodon and a NAUTILUS box sit at the
   same place (IoU >= ``--fuse-iou``) they collapse into one box, chosen by
   ``--fuse``. Roughly a third of NAUTILUS's boxes have a Megalodon partner, so
   this switch is what the parameter earns.
3. Each GT box **adopts** the geometry of a proposal that sits inside it, keeping
   its own label. "Inside" is *containment* -- the fraction of the **proposal's**
   area that falls within the GT box -- not IoU, because the GT box is known to be
   inflated and IoU would preferentially pick proposals that are also too big.
   Measured on the test split with Megalodon at conf 0.01, 80.5% of GT boxes have a
   proposal >=70% contained; the rest keep their original box unchanged.
4. Surviving proposals become **new annotations** under ``--unmatched-label``.

Coordinates are the step that silently breaks. Megalodon writes original-image
pixels; NAUTILUS writes model-input pixels, and only ``metadata.json`` knows the
scale. Everything here is normalised to ``[0, 1]`` on load -- ``x / input_width``,
which is exactly ``evaluate_detections.py``'s ``x * (ori_w / input_width) / ori_w``
-- so no comparison ever happens between two different pixel spaces.

Torch-free: pure geometry and json, so it runs on the host as well as in the
container.

Usage:
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/merge_annotations.py \
        --dataset        /workspace/datasets/thuenen_scaling \
        --megalodon-run  /workspace/runs/megalodon_proposals \
        --nautilus-run   /workspace/runs/nautilus_proposals \
        --out            /workspace/datasets/thuenen_refined \
        --megalodon-conf 0.25 --containment 0.7 --fuse megalodon

    # inspect the numbers before materialising anything:
    python3 merge_annotations.py ... --splits test --dry-run
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nautilus_zeroshot import link_or_copy  # noqa: E402

DEFAULT_DATASET = "/workspace/datasets/thuenen_scaling"
DEFAULT_MEGALODON = "/workspace/runs/megalodon_proposals"
DEFAULT_NAUTILUS = "/workspace/runs/nautilus_proposals"
DEFAULT_OUT = "/workspace/datasets/thuenen_refined"
DEFAULT_SPLITS = "train,val,test"
# Present in both label spaces already ("Unidentified organism" in classes.txt),
# so adding boxes under it needs no change to either class list.
DEFAULT_UNMATCHED_LABEL = "unidentified organism"
COPY_ALONGSIDE = ("classes.txt", "classes_prompt.txt", "prompt_names.csv", "splits.json")


# --------------------------------------------------------------------------- #
# geometry -- everything below operates on normalised (x1, y1, x2, y2)
# --------------------------------------------------------------------------- #

def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def iou(a, b):
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer):
    """Fraction of ``inner``'s own area that lies inside ``outer``.

    The asymmetry is the point: an inflated GT box contains a tight proposal
    almost entirely (containment ~1) while their IoU stays near 0.5.
    """
    inner_area = area(inner)
    return intersection_area(inner, outer) / inner_area if inner_area > 0 else 0.0


def combine_boxes(a, b, mode):
    if mode == "mean":
        return tuple((x + y) / 2.0 for x, y in zip(a, b))
    if mode == "union":
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
    if mode == "intersection":
        return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    raise ValueError("unknown combine mode {!r}".format(mode))


def clamp(box):
    return (min(max(box[0], 0.0), 1.0), min(max(box[1], 0.0), 1.0),
            min(max(box[2], 0.0), 1.0), min(max(box[3], 0.0), 1.0))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def read_class_names(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def find_class_id(names, label):
    """Case-insensitive lookup; the two label spaces disagree on capitalisation."""
    wanted = label.strip().lower()
    for index, name in enumerate(names):
        if name.strip().lower() == wanted:
            return index
    raise SystemExit("{!r} is not in the class list: {}".format(label, names))


def load_gt(labels_path, prompt_path):
    """Both label spaces at once.

    ``labels/`` and ``labels_prompt/`` are written from the same boxes in the same
    order by ``nautilus_zeroshot.write_split`` -- verified line-aligned with
    identical geometry across every file -- so one read yields both class ids per
    box. The alignment is asserted rather than assumed.
    """
    def rows(path):
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 5:
                    out.append((int(parts[0]), tuple(float(v) for v in parts[1:])))
        return out

    fine, prompt = rows(labels_path), rows(prompt_path)
    if len(fine) != len(prompt):
        raise SystemExit("label spaces disagree for {}: {} vs {} boxes".format(
            labels_path, len(fine), len(prompt)))

    boxes = []
    for (fine_id, geom), (prompt_id, prompt_geom) in zip(fine, prompt):
        if any(abs(a - b) > 1e-6 for a, b in zip(geom, prompt_geom)):
            raise SystemExit("label spaces disagree on geometry for " + labels_path)
        cx, cy, width, height = geom
        boxes.append({
            "fine_id": fine_id,
            "prompt_id": prompt_id,
            "box": (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        })
    return boxes


def load_proposals(run_dir, stem, dims, source, min_score):
    """One results file -> normalised, label-free proposals.

    ``metadata.json``'s ``input_*`` is the pixel space the boxes were written in
    (original size for Megalodon, model-input size for NAUTILUS), so dividing by it
    lands both on the same normalised frame.
    """
    path = os.path.join(run_dir, "results", stem + ".txt")
    if not os.path.exists(path) or dims is None:
        return []

    width, height = float(dims["input_width"]), float(dims["input_height"])
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "bbox_2d" not in record:
                continue
            score = record.get("score")
            if score is not None and score < min_score:
                continue
            x1, y1, x2, y2 = record["bbox_2d"][:4]
            box = clamp((min(x1, x2) / width, min(y1, y2) / height,
                         max(x1, x2) / width, max(y1, y2) / height))
            if area(box) <= 0:
                continue
            out.append({"box": box, "score": score, "sources": (source,)})
    return out


def load_metadata(run_dir, split):
    path = os.path.join(run_dir, split, "metadata.json")
    if not os.path.exists(path):
        raise SystemExit("missing {} -- was build_proposals.py run for this split?"
                         .format(path))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["image_dims"]


def load_drop_regions(path):
    """Recurring-artifact regions to delete before anything else looks at a box.

    Accepts either ``{"regions": [...]}`` or a bare list. Each region is
    ``{"box": [x1, y1, x2, y2]}`` in **normalised** coordinates, optionally with
    ``splits`` (list) and ``containment`` (default 0.8); a proposal is dropped when
    that fraction of it falls inside the region.
    """
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    regions = data.get("regions", data) if isinstance(data, dict) else data
    out = []
    for region in regions:
        out.append({
            "box": tuple(float(v) for v in region["box"]),
            "splits": region.get("splits"),
            "containment": float(region.get("containment", 0.8)),
            "comment": region.get("comment", ""),
        })
    return out


def dropped_by_region(box, regions, split):
    for region in regions:
        if region["splits"] and split not in region["splits"]:
            continue
        if containment(box, region["box"]) >= region["containment"]:
            return True
    return False


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #

def fuse_sources(megalodon, nautilus, mode, combine_mode, fuse_iou):
    """Collapse co-located Megalodon/NAUTILUS boxes into one pool.

    Greedy highest-IoU-first one-to-one, the same shape as
    ``evaluate_detections.greedy_match``: a box can take part in at most one pair,
    and unpaired boxes pass through untouched.
    """
    pairs = []
    for m_index, m in enumerate(megalodon):
        for n_index, n in enumerate(nautilus):
            overlap = iou(m["box"], n["box"])
            if overlap >= fuse_iou:
                pairs.append((overlap, m_index, n_index))
    pairs.sort(key=lambda item: -item[0])

    used_m, used_n, pool = set(), set(), []
    for _overlap, m_index, n_index in pairs:
        if m_index in used_m or n_index in used_n:
            continue
        used_m.add(m_index)
        used_n.add(n_index)
        m, n = megalodon[m_index], nautilus[n_index]
        if mode == "megalodon":
            box = m["box"]
        elif mode == "nautilus":
            box = n["box"]
        else:
            box = combine_boxes(m["box"], n["box"], combine_mode)
        pool.append({"box": clamp(box), "score": m["score"],
                     "sources": ("megalodon", "nautilus")})

    pool.extend(m for index, m in enumerate(megalodon) if index not in used_m)
    pool.extend(n for index, n in enumerate(nautilus) if index not in used_n)
    return pool, len(used_m)


def refine_ground_truth(gt_boxes, pool, min_containment):
    """Give every GT box the geometry of a proposal that sits inside it.

    Candidates are filtered by containment, then ranked by **IoU with the GT box**.
    Ranking by containment alone would prefer the smallest fragment (a tiny box is
    trivially 100% contained); among boxes that genuinely sit inside the GT, the
    highest-IoU one is the one that actually explains it.

    Assignment is global and one-to-one, so two GT boxes cannot claim the same
    proposal -- the better-fitting pair wins and the other GT box falls through to
    its original geometry.

    Returns ``(annotations, consumed_proposal_indices)``.
    """
    candidates = []
    for gt_index, gt in enumerate(gt_boxes):
        for p_index, proposal in enumerate(pool):
            if containment(proposal["box"], gt["box"]) >= min_containment:
                candidates.append((iou(proposal["box"], gt["box"]), gt_index, p_index))
    candidates.sort(key=lambda item: -item[0])

    chosen, used_gt, used_p = {}, set(), set()
    for _overlap, gt_index, p_index in candidates:
        if gt_index in used_gt or p_index in used_p:
            continue
        used_gt.add(gt_index)
        used_p.add(p_index)
        chosen[gt_index] = p_index

    annotations = []
    for gt_index, gt in enumerate(gt_boxes):
        p_index = chosen.get(gt_index)
        if p_index is None:
            annotations.append({
                "fine_id": gt["fine_id"], "prompt_id": gt["prompt_id"],
                "box": gt["box"], "origin": "gt_unchanged",
                "sources": (), "score": None, "gt_box": gt["box"],
            })
        else:
            proposal = pool[p_index]
            annotations.append({
                "fine_id": gt["fine_id"], "prompt_id": gt["prompt_id"],
                "box": proposal["box"], "origin": "gt_refined",
                "sources": proposal["sources"], "score": proposal["score"],
                "gt_box": gt["box"],
            })
    return annotations, used_p


def add_unmatched(annotations, pool, used, policy, fine_id, prompt_id, dedupe_iou):
    """Every leftover proposal the policy allows becomes a new annotation."""
    if policy == "none":
        return []

    added = []
    emitted = [annotation["box"] for annotation in annotations]
    for index, proposal in enumerate(pool):
        if index in used:
            continue
        sources = proposal["sources"]
        if policy == "megalodon" and "megalodon" not in sources:
            continue
        if policy == "nautilus" and "nautilus" not in sources:
            continue
        if any(iou(proposal["box"], other) >= dedupe_iou for other in emitted):
            continue
        emitted.append(proposal["box"])
        added.append({
            "fine_id": fine_id, "prompt_id": prompt_id, "box": proposal["box"],
            "origin": "added", "sources": sources, "score": proposal["score"],
            "gt_box": None,
        })
    return added


def to_yolo(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def format_label_line(class_id, box):
    cx, cy, width, height = to_yolo(box)
    return "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, cx, cy, width, height)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def merge_split(args, split, unmatched_ids, drop_regions):
    """Returns ``(stats, provenance)`` and, unless --dry-run, writes the split."""
    images_dir = os.path.join(args.dataset, split, "images")
    labels_dir = os.path.join(args.dataset, split, "labels")
    prompt_dir = os.path.join(args.dataset, split, "labels_prompt")

    megalodon_dims = load_metadata(args.megalodon_run, split)
    nautilus_dims = load_metadata(args.nautilus_run, split)
    megalodon_run = os.path.join(args.megalodon_run, split)
    nautilus_run = os.path.join(args.nautilus_run, split)

    if not args.dry_run:
        for name in ("images", "labels", "labels_prompt"):
            os.makedirs(os.path.join(args.out, split, name), exist_ok=True)

    stats = Counter()
    gt_areas, refined_areas, all_gt_areas, missing = [], [], [], []
    provenance = {}

    image_names = sorted(name for name in os.listdir(images_dir)
                         if os.path.splitext(name)[1].lower() in
                         (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))

    for image_name in image_names:
        stem = os.path.splitext(image_name)[0]
        stats["images"] += 1
        if image_name not in megalodon_dims or image_name not in nautilus_dims:
            missing.append(image_name)

        gt_boxes = load_gt(os.path.join(labels_dir, stem + ".txt"),
                           os.path.join(prompt_dir, stem + ".txt"))
        megalodon = load_proposals(megalodon_run, stem,
                                   megalodon_dims.get(image_name), "megalodon",
                                   args.megalodon_conf)
        nautilus = load_proposals(nautilus_run, stem,
                                  nautilus_dims.get(image_name), "nautilus", 0.0)
        stats["megalodon_proposals"] += len(megalodon)
        stats["nautilus_proposals"] += len(nautilus)

        if drop_regions:
            before = len(megalodon) + len(nautilus)
            megalodon = [p for p in megalodon
                         if not dropped_by_region(p["box"], drop_regions, split)]
            nautilus = [p for p in nautilus
                        if not dropped_by_region(p["box"], drop_regions, split)]
            stats["dropped_by_region"] += before - len(megalodon) - len(nautilus)

        pool, fused = fuse_sources(megalodon, nautilus, args.fuse,
                                   args.combine_mode, args.fuse_iou)
        stats["fused_pairs"] += fused

        annotations, used = refine_ground_truth(gt_boxes, pool, args.containment)
        added = add_unmatched(annotations, pool, used, args.unmatched,
                              unmatched_ids[0], unmatched_ids[1], args.dedupe_iou)
        annotations.extend(added)

        all_gt_areas.extend(area(gt["box"]) for gt in gt_boxes)
        stats["gt_boxes"] += len(gt_boxes)
        stats["added"] += len(added)
        for annotation in annotations:
            stats[annotation["origin"]] += 1
            if annotation["origin"] == "gt_refined":
                stats["source_" + "+".join(annotation["sources"])] += 1
                gt_areas.append(area(annotation["gt_box"]))
                refined_areas.append(area(annotation["box"]))

        if not args.dry_run:
            link_or_copy(os.path.join(images_dir, image_name),
                         os.path.join(args.out, split, "images", image_name))
            for directory, key in (("labels", "fine_id"), ("labels_prompt", "prompt_id")):
                lines = [format_label_line(a[key], a["box"]) for a in annotations]
                target = os.path.join(args.out, split, directory, stem + ".txt")
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + ("\n" if lines else ""))

        if args.report:
            provenance[image_name] = [{
                "fine_id": a["fine_id"], "prompt_id": a["prompt_id"],
                "box": [round(v, 6) for v in a["box"]], "origin": a["origin"],
                "sources": list(a["sources"]), "score": a["score"],
                "gt_box": [round(v, 6) for v in a["gt_box"]] if a["gt_box"] else None,
            } for a in annotations]

    def median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    summary = dict(stats)
    summary["refined_fraction"] = (stats["gt_refined"] / stats["gt_boxes"]
                                   if stats["gt_boxes"] else 0.0)
    summary["median_gt_area_all"] = median(all_gt_areas)
    summary["median_area_refined_boxes"] = median(refined_areas)
    summary["median_area_their_gt_boxes"] = median(gt_areas)
    summary["median_area_ratio_refined_over_gt"] = median(
        [r / g for r, g in zip(refined_areas, gt_areas) if g > 0])
    summary["images_missing_proposals"] = len(missing)
    summary["boxes_out"] = stats["gt_unchanged"] + stats["gt_refined"] + stats["added"]
    return summary, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--megalodon-run", default=DEFAULT_MEGALODON)
    parser.add_argument("--nautilus-run", default=DEFAULT_NAUTILUS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)

    parser.add_argument("--megalodon-conf", type=float, default=0.25,
                        help="Operating point chosen by eyeballing the proposals.")
    parser.add_argument("--containment", type=float, default=0.7,
                        help="Fraction of a proposal that must lie inside a GT box "
                             "for the GT box to adopt its geometry.")
    parser.add_argument("--fuse", default="megalodon",
                        choices=["megalodon", "nautilus", "combine"],
                        help="Which box wins where the two models overlap.")
    parser.add_argument("--combine-mode", default="mean",
                        choices=["mean", "union", "intersection"],
                        help="--fuse combine: per-corner mean, enclosing box, or "
                             "overlap. mean by default -- union would re-inflate "
                             "exactly what this script exists to shrink.")
    parser.add_argument("--fuse-iou", type=float, default=0.5,
                        help="IoU at which two proposals count as the same object.")
    parser.add_argument("--unmatched", default="megalodon",
                        choices=["megalodon", "nautilus", "both", "none"],
                        help="Which leftover proposals become new annotations.")
    parser.add_argument("--unmatched-label", default=DEFAULT_UNMATCHED_LABEL)
    parser.add_argument("--dedupe-iou", type=float, default=0.5,
                        help="Suppress an added box overlapping an emitted one.")
    parser.add_argument("--drop-boxes", default=None,
                        help="JSON of normalised regions holding recurring artifacts.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report without writing the dataset.")
    parser.add_argument("--report", action="store_true",
                        help="Also write per-box provenance.json.")
    args = parser.parse_args()

    fine_names = read_class_names(os.path.join(args.dataset, "classes.txt"))
    prompt_names = read_class_names(os.path.join(args.dataset, "classes_prompt.txt"))
    unmatched_ids = (find_class_id(fine_names, args.unmatched_label),
                     find_class_id(prompt_names, args.unmatched_label))
    print("added boxes go in as {!r} (classes.txt id {}, classes_prompt.txt id {})"
          .format(args.unmatched_label, unmatched_ids[0], unmatched_ids[1]))

    drop_regions = load_drop_regions(args.drop_boxes)
    if drop_regions:
        print("{} artifact region(s) from {}".format(len(drop_regions), args.drop_boxes))

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        for name in COPY_ALONGSIDE:
            source = os.path.join(args.dataset, name)
            if os.path.exists(source):
                shutil.copyfile(source, os.path.join(args.out, name))

    report = {"config": {k: v for k, v in vars(args).items()}, "splits": {}}
    provenance_all = {}
    for split in splits:
        print("\n=== {} ===".format(split))
        summary, provenance = merge_split(args, split, unmatched_ids, drop_regions)
        report["splits"][split] = summary
        provenance_all[split] = provenance
        print("  {images} images, {gt_boxes} GT boxes".format(**summary))
        print("  refined {gt_refined} ({refined_fraction:.1%}), "
              "unchanged {gt_unchanged}, added {added} -> {boxes_out} boxes"
              .format(**summary))
        if summary["median_area_ratio_refined_over_gt"] is not None:
            print("  median refined/GT area ratio {:.3f} "
                  "(median relative area {:.5f} -> {:.5f})".format(
                      summary["median_area_ratio_refined_over_gt"],
                      summary["median_area_their_gt_boxes"],
                      summary["median_area_refined_boxes"]))
        print("  fused pairs {fused_pairs}, proposals in: "
              "megalodon {megalodon_proposals}, nautilus {nautilus_proposals}"
              .format(**summary))
        if summary.get("dropped_by_region"):
            print("  {} proposals dropped by artifact regions"
                  .format(summary["dropped_by_region"]))
        if summary["images_missing_proposals"]:
            print("  [warn] {} images had no proposal metadata"
                  .format(summary["images_missing_proposals"]))

    destination = args.out if not args.dry_run else "."
    if not args.dry_run or args.report:
        os.makedirs(destination, exist_ok=True)
        path = os.path.join(destination, "merge_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nreport: {}".format(path))
        if args.report:
            path = os.path.join(destination, "provenance.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(provenance_all, handle, indent=2)
            print("provenance: {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
