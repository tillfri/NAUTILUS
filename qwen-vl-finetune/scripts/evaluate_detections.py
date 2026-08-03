"""
evaluate_detections.py — Quantitative detection metrics for NAUTILUS vs. YOLO-format
ground truth.

Given:
  * a directory of YOLO-format ground-truth label files (`<image_stem>.txt`, lines
    `class_id cx cy w h` normalized to [0, 1]) — same format consumed by
    `visualize_detections.py`'s `load_yolo_gt()`,
  * a directory of raw NAUTILUS model-output text files (`<image_stem>.txt`, one per
    image, produced by `batch_inference.py`'s `results/` folder), and
  * the `metadata.json` written alongside those results (holds per-image
    `input_width`/`input_height` needed to rescale predicted boxes from Qwen's
    dynamically-resized model-input space back to original image space),

this script computes:

  1. Class-agnostic detection quality: mAP@0.5 and mAP@0.5:0.95, ignoring predicted/
     ground-truth labels entirely (pure localization quality).
  2. Class-aware detection quality: mAP@0.5 and mAP@0.5:0.95 computed per class and
     averaged (standard "mAP" definition), plus a full per-class breakdown
     (AP@0.5, AP@0.5:0.95, Precision@0.5, Recall@0.5, F1@0.5, support).
  3. A label confusion matrix over boxes that spatially matched (IoU > 0.3,
     class-agnostic one-to-one assignment) but where predicted label != GT label —
     useful for separating localization errors from classification errors.

Important simplification — no confidence scores:
    NAUTILUS text output does not carry a per-detection confidence score (unlike
    typical object detectors). Every prediction is therefore treated as equal
    confidence, which collapses the precision-recall curve to a single operating
    point at each IoU threshold. Applying the standard COCO/VOC AP interpolation to
    a single PR point reduces exactly to:

        AP_t = Precision_t * Recall_t      (at IoU threshold t)

    This is what this script computes. It is NOT directly comparable to mAP numbers
    from confidence-ranked detectors (e.g. YOLO, Faster R-CNN) evaluated with
    pycocotools, since those integrate over a full PR curve.

Matching algorithm:
    Boxes are matched per-image via greedy, highest-IoU-first, one-to-one
    assignment (a prediction and a ground-truth box are each used in at most one
    match). This is an approximation of the optimal (Hungarian) assignment but is
    standard practice and matches the description "every bbox in the ground truth
    can only be assigned to one bbox in the prediction".

Usage:
    python evaluate_detections.py \
        --gt-dir        /path/to/yolo_labels \
        --pred-dir      /path/to/batch_inference_output/results \
        --metadata-json /path/to/batch_inference_output/metadata.json \
        --image-dir     /path/to/original/images \
        --classes-file  /path/to/classes.txt \
        --save-json     /path/to/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision.ops import box_iou

# Reuse the already-implemented, well-tested parsing/rescaling helpers instead of
# duplicating them.
from visualize_detections import (
    extract_detections_from_text,
    find_image_for_stem,
    load_yolo_gt,
    rescale_bbox,
)

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50, 0.55, ..., 0.95
MATCH_ONLY_IOU = 0.3  # threshold used for the label-confusion / existence matching pass


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────
def greedy_match(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold, class_aware):
    """Greedy, highest-IoU-first, one-to-one box matching.

    Args:
        pred_boxes: list[[x1,y1,x2,y2]]
        pred_labels: list[str] (same length as pred_boxes)
        gt_boxes: list[[x1,y1,x2,y2]]
        gt_labels: list[str] (same length as gt_boxes)
        iou_threshold: minimum IoU to be eligible for a match
        class_aware: if True, a pred/gt pair may only match when labels are equal

    Returns:
        matched_pairs: list[(pred_idx, gt_idx, iou)]
        unmatched_pred_idxs: set[int]
        unmatched_gt_idxs: set[int]
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0 or n_gt == 0:
        return [], set(range(n_pred)), set(range(n_gt))

    pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
    ious = box_iou(pred_t, gt_t)  # [n_pred, n_gt]

    candidates = []
    for i in range(n_pred):
        for j in range(n_gt):
            iou_val = ious[i, j].item()
            if iou_val < iou_threshold:
                continue
            if class_aware and pred_labels[i] != gt_labels[j]:
                continue
            candidates.append((iou_val, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_pred, used_gt = set(), set()
    matched_pairs = []
    for iou_val, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched_pairs.append((i, j, iou_val))

    unmatched_pred = set(range(n_pred)) - used_pred
    unmatched_gt = set(range(n_gt)) - used_gt
    return matched_pairs, unmatched_pred, unmatched_gt


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(
    gt_dir: Path,
    pred_dir: Path,
    image_dir: Path,
    image_dims: dict,
    class_names: Optional[list],
    recursive: bool,
):
    """Build per-image {"gt": [...], "pred": [...]} records.

    Each box entry is {"bbox_2d": [x1,y1,x2,y2], "label": normalized_str}, in
    original image pixel space.
    """
    gt_stems = {p.stem for p in gt_dir.glob("*.txt")}
    pred_stems = {p.stem for p in pred_dir.glob("*.txt")}

    common_stems = sorted(gt_stems & pred_stems)
    gt_only = gt_stems - pred_stems
    pred_only = pred_stems - gt_stems

    for stem in sorted(gt_only):
        print(f"[warning] {stem}: ground truth present but no prediction file — skipping", file=sys.stderr)
    for stem in sorted(pred_only):
        print(f"[warning] {stem}: prediction present but no ground truth file — skipping", file=sys.stderr)

    dataset = {}
    num_missing_image = 0
    num_missing_metadata = 0

    for stem in common_stems:
        image_path = find_image_for_stem(image_dir, stem, recursive)
        if image_path is None:
            print(f"[warning] {stem}: no matching image found in {image_dir} — skipping", file=sys.stderr)
            num_missing_image += 1
            continue

        ori_w, ori_h = Image.open(image_path).size

        gt_path = gt_dir / f"{stem}.txt"
        gt_dets = load_yolo_gt(gt_path, ori_w, ori_h, class_names)
        for det in gt_dets:
            det["label"] = str(det["label"]).strip().lower()

        dims = image_dims.get(image_path.name)
        if dims is None:
            print(
                f"[warning] {stem}: no metadata.json entry for {image_path.name} — "
                f"cannot rescale predictions, skipping image (GT boxes counted as missed)",
                file=sys.stderr,
            )
            num_missing_metadata += 1
            dataset[stem] = {"gt": gt_dets, "pred": []}
            continue

        raw_text = (pred_dir / f"{stem}.txt").read_text()
        pred_dets = extract_detections_from_text(raw_text)

        inv_scale_w = ori_w / dims["input_width"]
        inv_scale_h = ori_h / dims["input_height"]
        for det in pred_dets:
            det["bbox_2d"] = rescale_bbox(det["bbox_2d"], inv_scale_w, inv_scale_h)
            det["label"] = str(det.get("label", "object")).strip().lower()

        dataset[stem] = {"gt": gt_dets, "pred": pred_dets}

    stats = {
        "num_images_evaluated": len(dataset),
        "num_images_gt_only_skipped": len(gt_only),
        "num_images_pred_only_skipped": len(pred_only),
        "num_images_missing_image_skipped": num_missing_image,
        "num_images_missing_metadata": num_missing_metadata,
    }
    return dataset, stats


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _boxes_and_labels(dets, label_filter=None):
    boxes, labels = [], []
    for d in dets:
        bbox = d.get("bbox_2d")
        if bbox is None or len(bbox) != 4:
            continue
        if label_filter is not None and d.get("label") != label_filter:
            continue
        boxes.append(bbox)
        labels.append(d.get("label"))
    return boxes, labels


def precision_recall_at(dataset, iou_threshold, class_aware, label_filter=None):
    """Aggregate TP/FP/FN across all images at one IoU threshold.

    If label_filter is given, only boxes of that class are considered (both GT and
    pred) — used for per-class metrics. class_aware controls whether matching also
    requires label equality (irrelevant when label_filter is set, since all boxes
    already share the same label, but kept for symmetry/clarity).
    """
    tp = fp = fn = 0
    for rec in dataset.values():
        pred_boxes, pred_labels = _boxes_and_labels(rec["pred"], label_filter)
        gt_boxes, gt_labels = _boxes_and_labels(rec["gt"], label_filter)

        matched_pairs, unmatched_pred, unmatched_gt = greedy_match(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold, class_aware
        )
        tp += len(matched_pairs)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn


def ap_sweep(dataset, class_aware, label_filter=None):
    """Compute AP_t = Precision_t * Recall_t for t in IOU_THRESHOLDS.

    Returns (ap_at_each_threshold: dict[float,float], ap50: float, ap50_95: float,
    precision50, recall50, tp50, fp50, fn50).
    """
    ap_per_t = {}
    precision50 = recall50 = 0.0
    tp50 = fp50 = fn50 = 0
    for t in IOU_THRESHOLDS:
        precision, recall, tp, fp, fn = precision_recall_at(dataset, t, class_aware, label_filter)
        ap_per_t[t] = precision * recall
        if abs(t - 0.5) < 1e-9:
            precision50, recall50, tp50, fp50, fn50 = precision, recall, tp, fp, fn

    ap50 = ap_per_t[0.5]
    ap50_95 = sum(ap_per_t.values()) / len(ap_per_t)
    return ap_per_t, ap50, ap50_95, precision50, recall50, tp50, fp50, fn50


def compute_class_agnostic_metrics(dataset):
    _, ap50, ap50_95, precision50, recall50, tp50, fp50, fn50 = ap_sweep(
        dataset, class_aware=False, label_filter=None
    )
    f1_50 = (
        2 * precision50 * recall50 / (precision50 + recall50)
        if (precision50 + recall50) > 0
        else 0.0
    )
    return {
        "mAP@0.5": ap50,
        "mAP@0.5:0.95": ap50_95,
        "Precision@0.5": precision50,
        "Recall@0.5": recall50,
        "F1@0.5": f1_50,
        "TP@0.5": tp50,
        "FP@0.5": fp50,
        "FN@0.5": fn50,
    }


def compute_class_aware_metrics(dataset):
    all_classes = set()
    for rec in dataset.values():
        for d in rec["gt"]:
            all_classes.add(d["label"])
        for d in rec["pred"]:
            all_classes.add(d["label"])
    all_classes = sorted(all_classes)

    per_class = {}
    for cls in all_classes:
        _, ap50, ap50_95, precision50, recall50, tp50, fp50, fn50 = ap_sweep(
            dataset, class_aware=True, label_filter=cls
        )
        f1_50 = (
            2 * precision50 * recall50 / (precision50 + recall50)
            if (precision50 + recall50) > 0
            else 0.0
        )
        support = sum(1 for rec in dataset.values() for d in rec["gt"] if d["label"] == cls)
        num_pred = sum(1 for rec in dataset.values() for d in rec["pred"] if d["label"] == cls)
        per_class[cls] = {
            "AP@0.5": ap50,
            "AP@0.5:0.95": ap50_95,
            "Precision@0.5": precision50,
            "Recall@0.5": recall50,
            "F1@0.5": f1_50,
            "support": support,
            "num_pred": num_pred,
        }

    if per_class:
        map50 = sum(v["AP@0.5"] for v in per_class.values()) / len(per_class)
        map50_95 = sum(v["AP@0.5:0.95"] for v in per_class.values()) / len(per_class)
    else:
        map50 = map50_95 = 0.0

    return {"mAP@0.5": map50, "mAP@0.5:0.95": map50_95}, per_class


def compute_label_confusion(dataset, iou_threshold=MATCH_ONLY_IOU):
    """Class-agnostic matching at iou_threshold; tally (gt_label -> pred_label)
    counts for matched pairs, plus overall matched/unmatched counts."""
    confusion = {}
    matched_total = unmatched_gt_total = unmatched_pred_total = 0

    for rec in dataset.values():
        pred_boxes, pred_labels = _boxes_and_labels(rec["pred"])
        gt_boxes, gt_labels = _boxes_and_labels(rec["gt"])

        matched_pairs, unmatched_pred, unmatched_gt = greedy_match(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold, class_aware=False
        )
        for pred_idx, gt_idx, _iou in matched_pairs:
            gt_label = gt_labels[gt_idx]
            pred_label = pred_labels[pred_idx]
            confusion.setdefault(gt_label, {}).setdefault(pred_label, 0)
            confusion[gt_label][pred_label] += 1

        matched_total += len(matched_pairs)
        unmatched_gt_total += len(unmatched_gt)
        unmatched_pred_total += len(unmatched_pred)

    return confusion, {
        "matched": matched_total,
        "unmatched_gt": unmatched_gt_total,
        "unmatched_pred": unmatched_pred_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(metrics: dict):
    print("\n" + "=" * 70)
    print("DETECTION EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nImages evaluated: {metrics['num_images_evaluated']}")
    print(f"  Skipped (GT only, no prediction file):  {metrics['num_images_gt_only_skipped']}")
    print(f"  Skipped (prediction only, no GT file):  {metrics['num_images_pred_only_skipped']}")
    print(f"  Skipped (no matching image found):      {metrics['num_images_missing_image_skipped']}")
    print(f"  Skipped/degraded (no metadata.json entry): {metrics['num_images_missing_metadata']}")

    ca = metrics["class_agnostic"]
    print("\n--- Class-agnostic (localization only) ---")
    print(f"  mAP@0.5:      {ca['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {ca['mAP@0.5:0.95']:.4f}")
    print(f"  Precision@0.5: {ca['Precision@0.5']:.4f}  Recall@0.5: {ca['Recall@0.5']:.4f}  F1@0.5: {ca['F1@0.5']:.4f}")
    print(f"  TP={ca['TP@0.5']} FP={ca['FP@0.5']} FN={ca['FN@0.5']}")

    cw = metrics["class_aware"]
    print("\n--- Class-aware ---")
    print(f"  mAP@0.5:      {cw['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {cw['mAP@0.5:0.95']:.4f}")

    print("\n--- Per-class ---")
    header = f"{'class':<20}{'AP@0.5':>10}{'AP@.5:.95':>12}{'P@0.5':>10}{'R@0.5':>10}{'F1@0.5':>10}{'support':>10}{'pred':>8}"
    print(header)
    print("-" * len(header))
    for cls, v in metrics["per_class"].items():
        print(
            f"{cls:<20}{v['AP@0.5']:>10.4f}{v['AP@0.5:0.95']:>12.4f}"
            f"{v['Precision@0.5']:>10.4f}{v['Recall@0.5']:>10.4f}{v['F1@0.5']:>10.4f}"
            f"{v['support']:>10}{v['num_pred']:>8}"
        )

    lc = metrics["localization_only_iou_0.3"]
    print(f"\n--- Localization-only matching (IoU > {MATCH_ONLY_IOU}, class-agnostic) ---")
    print(f"  Matched: {lc['matched']}  Unmatched GT: {lc['unmatched_gt']}  Unmatched Pred: {lc['unmatched_pred']}")

    print("\n--- Label confusion matrix (matched boxes @ IoU > 0.3) ---")
    print("  (rows = ground-truth label, columns = predicted label)")
    confusion = metrics["label_confusion_matrix_iou_0.3"]
    for gt_label, pred_counts in confusion.items():
        row = ", ".join(f"{pred_label}={count}" for pred_label, count in sorted(pred_counts.items()))
        print(f"  {gt_label:<20} {row}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-agnostic and class-aware detection metrics "
            "(mAP@0.5, mAP@0.5:0.95, per-class Precision/Recall/F1, label confusion) "
            "for NAUTILUS predictions against YOLO-format ground truth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gt-dir", required=True,
        help="Directory of YOLO-format ground-truth label files (<image_stem>.txt).",
    )
    parser.add_argument(
        "--pred-dir", required=True,
        help="Directory of raw NAUTILUS result .txt files (<image_stem>.txt), e.g. "
             "<batch_inference output-dir>/results.",
    )
    parser.add_argument(
        "--metadata-json", required=True,
        help="Path to metadata.json produced by batch_inference.py (image_dims with "
             "input_width/input_height per image, used to rescale predicted bboxes).",
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Directory containing the original images (for width/height lookup).",
    )
    parser.add_argument(
        "--classes-file", required=True,
        help="File with one class name per line (line index = YOLO class id). "
             "Required: without it, GT labels are numeric ids and cannot match "
             "NAUTILUS's free-text labels, making class-aware metrics meaningless.",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="Optional path to write the full metrics dict as JSON.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search --image-dir recursively.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    metadata_path = Path(args.metadata_json)

    for label, p in [("Ground-truth", gt_dir), ("Prediction", pred_dir), ("Image", image_dir)]:
        if not p.exists():
            print(f"[error] {label} directory not found: {p}", file=sys.stderr)
            sys.exit(1)
    if not metadata_path.exists():
        print(f"[error] metadata.json not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    class_names = [
        line.strip() for line in Path(args.classes_file).read_text().splitlines() if line.strip()
    ]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    image_dims = metadata.get("image_dims", {})
    if not image_dims:
        print(f"[warning] metadata.json at {metadata_path} has no 'image_dims' entries", file=sys.stderr)

    dataset, load_stats = load_dataset(
        gt_dir, pred_dir, image_dir, image_dims, class_names, args.recursive
    )

    if not dataset:
        print("[error] No images could be loaded — nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    class_agnostic = compute_class_agnostic_metrics(dataset)
    class_aware, per_class = compute_class_aware_metrics(dataset)
    confusion, loc_stats = compute_label_confusion(dataset)

    metrics = {
        **load_stats,
        "class_agnostic": class_agnostic,
        "class_aware": class_aware,
        "per_class": per_class,
        "label_confusion_matrix_iou_0.3": confusion,
        "localization_only_iou_0.3": loc_stats,
    }

    print_summary(metrics)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nFull metrics written to {save_path}")


if __name__ == "__main__":
    main()
