"""Box-convention-robust, class-agnostic localization metrics for the Thünen split.

``evaluate_detections.py`` answers "did the model reproduce this dataset?". This
script answers the narrower question "did the model *find the animal*?", which is
the only one that can be asked fairly across models that draw boxes to different
conventions.

Why it exists: a quarter of Thünen GT boxes are axis-aligned squares produced by
BIIGLE's circle tool, and GT boxes are ~2.09x the area of NAUTILUS's. A perfectly
tight box on the median target therefore scores IoU 1/2.09 = 0.478 -- below the
0.5 threshold -- and is counted as a false positive *and* a false negative at once.
Any single-IoU comparison between a model that learned the convention (YOLO, trained
on these labels) and one that did not (NAUTILUS, Megalodon) is measuring the
annotation tool as much as the model. So this reports:

* class-agnostic recall/precision at IoU 0.5, 0.4, 0.3, 0.2, 0.1 **and** at
  centre-in-box, which is scale-free and immune to the inflation;
* the same recalls split by GT box convention -- circle-derived (square) GT vs
  rectangle-drawn GT -- which is where the whole NAUTILUS-vs-YOLO reversal lives;
* per-GT-class class-agnostic recall, so a model that finds one taxon and no other
  is visible as such;
* box statistics (count, per image, empty-image fraction, median area, square
  fraction) to place a model's output budget next to the GT's.

Labels are ignored by every metric here -- ``--classes-file`` only supplies GT class
names for the per-class breakdown. That makes the output directly comparable across
a class-agnostic detector (Megalodon, one ``object`` class), a multi-class detector
(YOLOv8m, 29 classes) and a generative model (NAUTILUS, free text).

Two caveats carried from the notes: the square/rectangle split is a **proxy** (1:1
aspect +/-2%), not read back from the source BIIGLE CSVs; and Thünen annotations are
sparse rather than exhaustive, so precision is a lower bound -- an unmatched box is
not evidence of a false positive. Read recall against boxes-per-image.

Usage:
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
localization_report.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_megalodon_i1280_c050/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_megalodon_i1280_c050/localization.json
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from evaluate_detections import greedy_match, load_dataset
from visualize_detections import find_image_for_stem, load_metadata

IOU_CRITERIA = [0.5, 0.4, 0.3, 0.2, 0.1]
SQUARE_TOLERANCE = 0.02  # |w/h - 1| <= this counts as "exactly square"


def is_square(bbox):
    """Proxy for 'this GT box came from BIIGLE's circle tool'."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return False
    return abs(width / height - 1.0) <= SQUARE_TOLERANCE


def centre_in_box_match(pred_boxes, gt_boxes):
    """Greedy one-to-one match on 'the prediction's centre falls inside the GT box'.

    Scale-free: unlike IoU it does not care that GT boxes are ~2x the area of a
    tight box. Ties are resolved by centre distance, so the closest prediction wins
    a contested GT box, mirroring greedy_match's highest-IoU-first ordering.
    """
    candidates = []
    for i, pred in enumerate(pred_boxes):
        cx = (pred[0] + pred[2]) / 2.0
        cy = (pred[1] + pred[3]) / 2.0
        for j, gt in enumerate(gt_boxes):
            if not (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]):
                continue
            gcx = (gt[0] + gt[2]) / 2.0
            gcy = (gt[1] + gt[3]) / 2.0
            candidates.append((((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5, i, j))

    candidates.sort(key=lambda c: c[0])
    used_pred, used_gt, matched = set(), set(), []
    for _, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched.append((i, j))
    return matched


def boxes_of(dets):
    return [d["bbox_2d"] for d in dets
            if isinstance(d.get("bbox_2d"), list) and len(d["bbox_2d"]) == 4]


def box_stats(dataset, key, image_sizes):
    """Count / density / median area / square fraction for the gt or pred side."""
    per_image, areas, squares, total = [], [], 0, 0
    for stem, record in dataset.items():
        bboxes = boxes_of(record[key])
        per_image.append(len(bboxes))
        total += len(bboxes)
        frame_w, frame_h = image_sizes[stem]
        frame_area = float(frame_w * frame_h) or 1.0
        for bbox in bboxes:
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / frame_area)
            squares += 1 if is_square(bbox) else 0
    n_images = len(per_image) or 1
    return {
        "boxes": total,
        "boxes_per_image": round(total / n_images, 4),
        "empty_images": sum(1 for c in per_image if c == 0),
        "empty_image_frac": round(sum(1 for c in per_image if c == 0) / n_images, 4),
        "median_area_frac": round(statistics.median(areas), 6) if areas else None,
        "square_boxes": squares,
        "square_frac": round(squares / total, 4) if total else None,
    }


def evaluate_criterion(dataset, mode, threshold=None):
    """One matching pass; recalls for every GT subset are derived from it.

    Matching is done once over all boxes in an image and the matched GT indices are
    then bucketed, rather than re-matching per subset -- re-matching a subset would
    let a prediction that legitimately claimed a neighbouring box be reused.
    """
    n_pred = n_gt = matched = 0
    subset = {"square": [0, 0], "rect": [0, 0]}  # [matched, total]
    per_class = {}

    for record in dataset.values():
        pred_boxes = boxes_of(record["pred"])
        gt_dets = [d for d in record["gt"]
                   if isinstance(d.get("bbox_2d"), list) and len(d["bbox_2d"]) == 4]
        gt_boxes = [d["bbox_2d"] for d in gt_dets]

        if mode == "iou":
            pairs, _, _ = greedy_match(pred_boxes, [None] * len(pred_boxes),
                                       gt_boxes, [None] * len(gt_boxes),
                                       threshold, class_aware=False)
            matched_gt = {j for _, j, _ in pairs}
        else:
            matched_gt = {j for _, j in centre_in_box_match(pred_boxes, gt_boxes)}

        n_pred += len(pred_boxes)
        n_gt += len(gt_boxes)
        matched += len(matched_gt)

        for j, det in enumerate(gt_dets):
            bucket = "square" if is_square(det["bbox_2d"]) else "rect"
            subset[bucket][1] += 1
            entry = per_class.setdefault(det["label"], [0, 0])
            entry[1] += 1
            if j in matched_gt:
                subset[bucket][0] += 1
                entry[0] += 1

    def ratio(num, den):
        return round(num / den, 4) if den else None

    recall = ratio(matched, n_gt)
    precision = ratio(matched, n_pred)
    f1 = None
    if recall and precision:
        f1 = round(2 * recall * precision / (recall + precision), 4)

    return {
        "matched": matched, "n_gt": n_gt, "n_pred": n_pred,
        "recall": recall, "precision": precision, "f1": f1,
        "square_gt": {"matched": subset["square"][0], "total": subset["square"][1],
                      "recall": ratio(*subset["square"])},
        "rect_gt": {"matched": subset["rect"][0], "total": subset["rect"][1],
                    "recall": ratio(*subset["rect"])},
        "per_class": {label: {"matched": m, "support": s, "recall": ratio(m, s)}
                      for label, (m, s) in sorted(per_class.items(),
                                                  key=lambda kv: -kv[1][1])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--classes-file", required=True, type=Path)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--label", default=None,
                        help="report label, defaults to the pred-dir's parent name")
    args = parser.parse_args()

    class_names = [line.strip() for line in
                   args.classes_file.read_text().splitlines() if line.strip()]
    image_dims = load_metadata(args.pred_dir.parent).get("image_dims", {})
    if not image_dims:
        raise SystemExit("no image_dims in {}/metadata.json".format(args.pred_dir.parent))

    dataset, stats = load_dataset(args.gt_dir, args.pred_dir, args.image_dir,
                                  image_dims, class_names, args.recursive)
    if not dataset:
        raise SystemExit("no images with both ground truth and predictions")

    image_sizes = {}
    for stem in dataset:
        image_path = find_image_for_stem(args.image_dir, stem, args.recursive)
        image_sizes[stem] = Image.open(image_path).size

    report = {
        "label": args.label or args.pred_dir.parent.name,
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "num_images_evaluated": stats["num_images_evaluated"],
        "load_stats": stats,
        "gt_boxes": box_stats(dataset, "gt", image_sizes),
        "pred_boxes": box_stats(dataset, "pred", image_sizes),
        "criteria": {},
    }
    for iou in IOU_CRITERIA:
        report["criteria"]["iou_{:.1f}".format(iou)] = evaluate_criterion(dataset, "iou", iou)
    report["criteria"]["centre_in_box"] = evaluate_criterion(dataset, "centre")

    print("\n=== {} ===".format(report["label"]))
    print("images evaluated: {}".format(report["num_images_evaluated"]))
    for side in ("gt_boxes", "pred_boxes"):
        s = report[side]
        print("{:<10} {:>6} boxes  {:>5.2f}/img  {:>5.1%} empty  median area {}  "
              "{:>5.1%} square".format(
                  side.replace("_boxes", ""), s["boxes"], s["boxes_per_image"],
                  s["empty_image_frac"],
                  "n/a" if s["median_area_frac"] is None else "{:.4%}".format(s["median_area_frac"]),
                  s["square_frac"] or 0.0))

    print("\n{:<16}{:>9}{:>11}{:>9}{:>13}{:>13}".format(
        "criterion", "recall", "precision", "f1", "recall sq GT", "recall rect GT"))
    for name, block in report["criteria"].items():
        print("{:<16}{:>9}{:>11}{:>9}{:>13}{:>13}".format(
            name,
            "n/a" if block["recall"] is None else "{:.4f}".format(block["recall"]),
            "n/a" if block["precision"] is None else "{:.4f}".format(block["precision"]),
            "n/a" if block["f1"] is None else "{:.4f}".format(block["f1"]),
            "{}/{}".format(block["square_gt"]["matched"], block["square_gt"]["total"]),
            "{}/{}".format(block["rect_gt"]["matched"], block["rect_gt"]["total"])))

    print("\nper-class recall @ IoU 0.3 (top 10 by support)")
    per_class = report["criteria"]["iou_0.3"]["per_class"]
    for label, entry in list(per_class.items())[:10]:
        print("  {:<26} {:>4}/{:<5} {:.3f}".format(
            label[:26], entry["matched"], entry["support"], entry["recall"] or 0.0))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("\nwrote {}".format(args.save_json))


if __name__ == "__main__":
    main()
