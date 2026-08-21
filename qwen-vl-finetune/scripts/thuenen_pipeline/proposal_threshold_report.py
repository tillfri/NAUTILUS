"""Sweep the Megalodon proposal confidence and report what each threshold buys.

Step 2 of the annotation-refinement plan: ``build_proposals.py --model megalodon``
was run once at ``conf 0.01`` so the threshold could be chosen afterwards as a free
in-memory filter. This script is that filter. It reads
``runs/megalodon_proposals/<split>/{results,metadata.json}`` and, for a list of
thresholds, reports

* **density** -- boxes per image (mean/median/p90/max) and the share of images that
  keep at least one proposal, next to the ground truth's own boxes per image;
* **what the merge would do with them** -- how many GT boxes find a proposal that
  sits inside them at ``--containment`` (those get tightened geometry), and how many
  proposals are left over to become *new* annotations under ``--unmatched megalodon``
  (deduped exactly as ``merge_annotations.py`` dedupes them).

The second block is the one that decides the threshold, and it says the two uses
want opposite values: coverage never saturates (it climbs all the way to conf 0.01)
while added boxes per image explode. ``merge_annotations.py`` therefore takes two
thresholds -- read ``GT ref.``/``ref%`` off the ``--megalodon-conf`` row and
``added``/``add/img`` off the (higher) ``--megalodon-add-conf`` row; each column is
independent of the other threshold.

The matching, dedupe and normalisation code is imported from ``merge_annotations``
rather than reimplemented, so the numbers here are the numbers that merge produces.
Torch-free, so it also runs on the host.

Usage:
    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \
       python3 proposal_threshold_report.py \
         --run /workspace/runs/megalodon_proposals \
         --dataset /workspace/datasets/thuenen_scaling \
         --splits train,val,test \
         --save-json /workspace/runs/megalodon_proposals/threshold_report.json"

    # density only, no ground truth, host-side:
    python3 proposal_threshold_report.py --run ~/nautilus/runs/megalodon_proposals \
        --splits test --no-gt

    # include the NAUTILUS proposals in the pool (fused as merge_annotations.py
    # fuses them, --fuse megalodon), to see the coverage the merge really gets:
    python3 proposal_threshold_report.py --nautilus-run ~/nautilus/runs/nautilus_proposals
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from merge_annotations import (  # noqa: E402
    add_unmatched, area, fuse_sources, load_gt, load_metadata, load_proposals,
    refine_ground_truth,
)

DEFAULT_RUN = "/workspace/runs/megalodon_proposals"
DEFAULT_DATASET = "/workspace/datasets/thuenen_scaling"
DEFAULT_SPLITS = "train,val,test"
DEFAULT_THRESHOLDS = "0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7"


def percentile(values, fraction):
    """Nearest-rank percentile of an unsorted list; 0.0 for an empty one."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def median(values):
    return percentile(values, 0.5)


def load_split(args, split):
    """Everything one split needs, read from disk exactly once.

    Proposals are loaded at the lowest requested threshold and filtered per
    threshold in memory -- the whole reason the detector ran at conf 0.01.
    """
    dims = load_metadata(args.run, split)
    run_dir = os.path.join(args.run, split)
    nautilus_dims = load_metadata(args.nautilus_run, split) if args.nautilus_run else {}
    nautilus_dir = os.path.join(args.nautilus_run, split) if args.nautilus_run else None

    labels_dir = os.path.join(args.dataset, split, "labels")
    prompt_dir = os.path.join(args.dataset, split, "labels_prompt")

    images = sorted(dims)
    if args.limit:
        images = images[:args.limit]

    records = []
    for name in images:
        stem = os.path.splitext(name)[0]
        proposals = load_proposals(run_dir, stem, dims.get(name), "megalodon", 0.0)
        nautilus = []
        if nautilus_dir is not None:
            nautilus = load_proposals(nautilus_dir, stem, nautilus_dims.get(name),
                                      "nautilus", 0.0)
        gt = []
        if not args.no_gt:
            gt = load_gt(os.path.join(labels_dir, stem + ".txt"),
                         os.path.join(prompt_dir, stem + ".txt"))
        records.append({"proposals": proposals, "nautilus": nautilus, "gt": gt})
    return records


def evaluate_threshold(args, records, threshold):
    """One row of the sweep: density plus the merge outcome at this threshold."""
    per_image, areas = [], []
    gt_total = gt_refined = added_total = 0
    area_ratios = []

    for record in records:
        kept = [p for p in record["proposals"]
                if p["score"] is None or p["score"] >= threshold]
        per_image.append(len(kept))
        areas.extend(area(p["box"]) for p in kept)

        if args.no_gt:
            continue

        pool = kept
        if record["nautilus"]:
            pool, _fused = fuse_sources(kept, record["nautilus"], "megalodon",
                                        "mean", args.fuse_iou)
        gt_boxes = record["gt"]
        gt_total += len(gt_boxes)

        annotations, used = refine_ground_truth(gt_boxes, pool, args.containment)
        for annotation in annotations:
            if annotation["origin"] == "gt_refined":
                gt_refined += 1
                gt_area = area(annotation["gt_box"])
                if gt_area > 0:
                    area_ratios.append(area(annotation["box"]) / gt_area)

        added_total += len(add_unmatched(annotations, pool, used, args.unmatched,
                                         0, 0, args.dedupe_iou))

    images = len(per_image) or 1
    boxes = sum(per_image)
    row = {
        "conf": threshold,
        "boxes": boxes,
        "boxes_per_image": boxes / images,
        "median_boxes": median(per_image),
        "p90_boxes": percentile(per_image, 0.9),
        "max_boxes": max(per_image) if per_image else 0,
        "images_with_box_frac": sum(1 for n in per_image if n) / images,
        "median_box_area": median(areas),
    }
    if not args.no_gt:
        row.update({
            "gt_boxes": gt_total,
            "gt_refined": gt_refined,
            "gt_refined_frac": gt_refined / gt_total if gt_total else 0.0,
            "median_refined_area_ratio": median(area_ratios),
            "added_boxes": added_total,
            "added_per_image": added_total / images,
        })
    return row


HEADER_DENSITY = ("  conf   boxes  box/img   p50   p90   max  img>=1  "
                  "med.area")
HEADER_GT = "  GT ref.   ref%  area/GT   added  add/img  boxes/img after"


def print_split(split, records, rows, with_gt):
    images = len(records)
    gt_boxes = sum(len(record["gt"]) for record in records)
    print()
    print("=" * len(HEADER_DENSITY + (HEADER_GT if with_gt else "")))
    header = "{}  --  {} images".format(split, images)
    if with_gt:
        header += ", {} GT boxes ({:.2f} box/img)".format(gt_boxes, gt_boxes / max(1, images))
    print(header)
    print("=" * len(HEADER_DENSITY + (HEADER_GT if with_gt else "")))
    print(HEADER_DENSITY + (HEADER_GT if with_gt else ""))
    for row in rows:
        line = ("{conf:6.3f} {boxes:7d} {boxes_per_image:8.2f} {median_boxes:5.0f} "
                "{p90_boxes:5.0f} {max_boxes:5d} {images_with_box_frac:6.1%} "
                "{median_box_area:9.5f}").format(**row)
        if with_gt:
            line += ("  {gt_refined:7d} {gt_refined_frac:6.1%} "
                     "{median_refined_area_ratio:8.3f} {added_boxes:7d} "
                     "{added_per_image:8.2f}").format(**row)
            after = (row["gt_boxes"] + row["added_boxes"]) / max(1, images)
            line += "  {:14.2f}".format(after)
        print(line)


def print_marginal(split, rows):
    """What each step down in threshold costs: added boxes bought per GT box refined.

    Coverage has no knee -- it climbs all the way to conf 0.01 -- so the decision is
    a price, not an inflection: below ~0.1 a refined GT box costs >20 new boxes.
    """
    print()
    print("{}  --  marginal yield of lowering the threshold".format(split))
    print("  band            +refined   +added   refined per 100 added")
    for upper, lower in zip(rows[1:], rows[:-1]):
        refined = lower["gt_refined"] - upper["gt_refined"]
        added = lower["added_boxes"] - upper["added_boxes"]
        print("  {:.3f} -> {:.3f}  {:8d} {:8d}   {:20.1f}".format(
            upper["conf"], lower["conf"], refined, added,
            100.0 * refined / added if added else 0.0))


def print_legend(with_gt):
    print()
    print("box/img  proposals kept per image        med.area  median proposal area, "
          "normalised")
    print("img>=1   images keeping >=1 proposal")
    if with_gt:
        print("GT ref.  GT boxes that get proposal geometry at containment >= "
              "the --containment threshold  (read at --megalodon-conf)")
        print("area/GT  median (refined box area / original GT box area) -- "
              "how much the GT shrinks")
        print("added    leftover proposals that become new annotations "
              "(--unmatched policy, deduped)  (read at --megalodon-add-conf)")
        print("boxes/img after  (GT boxes + added) / images -- the density of the "
              "refined dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="megalodon proposal run root (contains <split>/results)")
    parser.add_argument("--nautilus-run", default=None,
                        help="optionally fuse NAUTILUS proposals into the pool, as "
                             "merge_annotations.py --fuse megalodon would")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--containment", type=float, default=0.7)
    parser.add_argument("--fuse-iou", type=float, default=0.5)
    parser.add_argument("--dedupe-iou", type=float, default=0.5)
    parser.add_argument("--unmatched", default="megalodon",
                        choices=("megalodon", "both", "nautilus", "none"))
    parser.add_argument("--marginal", action="store_true",
                        help="also print what each step down in threshold costs")
    parser.add_argument("--no-gt", action="store_true",
                        help="density only; skip the ground-truth interaction")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N images of each split (debugging)")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    thresholds = sorted(float(value) for value in args.thresholds.split(",") if value)
    splits = [value for value in args.splits.split(",") if value]
    with_gt = not args.no_gt

    report = {
        "run": args.run,
        "nautilus_run": args.nautilus_run,
        "dataset": None if args.no_gt else args.dataset,
        "containment": args.containment,
        "unmatched": args.unmatched,
        "dedupe_iou": args.dedupe_iou,
        "splits": {},
    }

    for split in splits:
        records = load_split(args, split)
        rows = [evaluate_threshold(args, records, threshold) for threshold in thresholds]
        print_split(split, records, rows, with_gt)
        if args.marginal and with_gt:
            print_marginal(split, rows)
        report["splits"][split] = {
            "images": len(records),
            "gt_boxes": sum(len(record["gt"]) for record in records),
            "thresholds": rows,
        }

    print_legend(with_gt)

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nwrote " + args.save_json)


if __name__ == "__main__":
    main()
