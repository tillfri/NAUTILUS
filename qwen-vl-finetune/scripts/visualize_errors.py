"""
visualize_errors.py — Render only the images NAUTILUS got wrong, with the error types
split into separate panels.

`evaluate_detections.py` reports aggregate numbers (mAP, per-class P/R/F1, TP/FP/FN)
but not *which* images failed. `visualize_detections.py` renders a GT-vs-prediction
pair for every image, including the many that are perfectly correct. This script sits
in between: it runs the same matching logic as the evaluator, keeps only the images
that contain at least one false positive or false negative, and draws each one as a
2x2 figure:

    ┌──────────────────────┬──────────────────────┐
    │ Ground Truth         │ True Positives       │
    │ (all GT boxes)       │ (matched predictions,│
    │                      │  captioned with IoU) │
    ├──────────────────────┼──────────────────────┤
    │ False Positives      │ False Negatives      │
    │ (unmatched preds,    │ (unmatched GT boxes) │
    │  captioned with      │                      │
    │  score if present)   │                      │
    └──────────────────────┴──────────────────────┘

Matching is class-aware by default: a prediction only counts as a true positive when
IoU >= --iou-threshold *and* its label equals the ground-truth label. A well-localized
box carrying the wrong label therefore shows up twice — as a false positive in the
lower-left tile and as a false negative in the lower-right tile, at the same position.
Pass --class-agnostic to ignore labels entirely and surface only pure misses and
hallucinations.

All parsing, rescaling and matching is reused from the two sibling scripts, so the
numbers here are directly comparable to `evaluate_detections.py`'s output.

Usage:
    python visualize_errors.py \
        --gt-dir       /path/to/yolo_labels \
        --pred-dir     /path/to/batch_inference_output/results \
        --image-dir    /path/to/original/images \
        --classes-file /path/to/classes.txt \
        [--iou-threshold 0.5] [--class-agnostic] [--fishies] [--limit 50]

    `metadata.json` is always read from `<pred-dir>/../metadata.json`, matching
    `evaluate_detections.py`.

Arguments:
    --gt-dir          Directory of YOLO-format ground-truth label files (<stem>.txt).
    --pred-dir        Directory of raw NAUTILUS result .txt files (the `results/`
                      folder written by batch_inference.py).
    --image-dir       Directory containing the original images.
    --classes-file    File with one class name per line (line index = YOLO class id).
    --save-dir        Where to write the 2x2 figures
                      (default: <pred-dir>/../error_visualizations).
    --save-json       Where to write the machine-readable error report
                      (default: <save-dir>/errors.json).
    --iou-threshold   IoU required for a prediction to match a GT box (default 0.5).
    --class-agnostic  Ignore labels when matching (default: class-aware).
    --fishies         Merge "small_fish" into "fish" before matching.
    --recursive       Search --image-dir recursively.
    --limit           Only write figures for the N worst images (most FP+FN). The
                      JSON report and the printed totals always cover every image.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluate_detections import _boxes_and_labels, greedy_match, load_dataset
from visualize_detections import _draw_panel, find_image_for_stem

matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Error splitting
# ─────────────────────────────────────────────────────────────────────────────
def split_errors(record: dict, iou_threshold: float, class_aware: bool):
    """Split one image's detections into true positives, false positives, false negatives.

    Args:
        record: {"gt": [det, ...], "pred": [det, ...]} as built by load_dataset().
        iou_threshold: minimum IoU for a prediction to match a ground-truth box.
        class_aware: if True, a pair may only match when the labels are equal.

    Returns:
        (tp, fp, fn) — three lists of detection dicts in original image pixel space.
        Each TP entry is the *predicted* box, with an extra "iou" field and a
        "display" caption of the form "<label> IoU=0.72". Each FP entry carries the
        predicted box's "score" (when the result file has one — NAUTILUS itself emits
        none, but e.g. YOLO-derived result files do) and a matching "display" caption
        of the form "<label> score=0.68".
    """
    pred_boxes, pred_labels = _boxes_and_labels(record["pred"])
    gt_boxes, gt_labels = _boxes_and_labels(record["gt"])
    # Same filter as _boxes_and_labels (valid bbox_2d of length 4), kept parallel so
    # index i lines up with pred_boxes[i]/pred_labels[i] for score lookup below.
    pred_scores = [
        d.get("score")
        for d in record["pred"]
        if isinstance(d.get("bbox_2d"), list) and len(d.get("bbox_2d")) == 4
    ]

    matched_pairs, unmatched_pred, unmatched_gt = greedy_match(
        pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold, class_aware
    )

    tp = []
    for pred_idx, _gt_idx, iou in matched_pairs:
        label = pred_labels[pred_idx]
        tp.append(
            {
                "bbox_2d": pred_boxes[pred_idx],
                "label": label,
                "iou": round(iou, 4),
                "display": f"{label} IoU={iou:.2f}",
            }
        )

    fp = []
    for i in sorted(unmatched_pred):
        label = pred_labels[i]
        score = pred_scores[i]
        entry = {"bbox_2d": pred_boxes[i], "label": label}
        if isinstance(score, (int, float)):
            entry["score"] = round(score, 4)
            entry["display"] = f"{label} score={score:.2f}"
        fp.append(entry)

    fn = [
        {"bbox_2d": gt_boxes[j], "label": gt_labels[j]}
        for j in sorted(unmatched_gt)
    ]
    return tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_error_figure(
    image_path: Path,
    gt_detections: list,
    tp: list,
    fp: list,
    fn: list,
    output_path: Path,
    stem: str,
    iou_threshold: float,
    class_aware: bool,
    no_pred_parsed: bool,
) -> None:
    """Render the 2x2 GT / TP / FP / FN comparison for one image."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Shape the figure to the image's aspect ratio — a fixed square figure leaves a
    # large empty band between the two rows for the usual wide underwater frames.
    img_h, img_w = img_array.shape[:2]
    fig_w = 20.0
    # + headroom for the suptitle and the two rows of panel titles
    fig_h = max(6.0, fig_w * (img_h / img_w) + 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    # One shared map so a given class keeps the same colour across all four tiles.
    label_map: dict = {}

    _draw_panel(axes[0][0], img_array, gt_detections, f"Ground Truth ({len(gt_detections)})", label_map)
    _draw_panel(axes[0][1], img_array, tp, f"True Positives ({len(tp)})", label_map, text_field="display")
    _draw_panel(axes[1][0], img_array, fp, f"False Positives ({len(fp)})", label_map, text_field="display")
    _draw_panel(axes[1][1], img_array, fn, f"False Negatives ({len(fn)})", label_map)

    if no_pred_parsed:
        axes[1][0].text(
            0.5, -0.03,
            "[no bbox parsed from model output]",
            transform=axes[1][0].transAxes,
            ha="center", va="top",
            fontsize=10, color="#cc0000",
        )

    mode = "class-aware" if class_aware else "class-agnostic"
    fig.suptitle(
        f"{stem} — TP={len(tp)} FP={len(fp)} FN={len(fn)}  "
        f"(IoU≥{iou_threshold:.2f}, {mode})",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(report: dict, load_stats: dict, num_figures: int, save_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("ERROR-CASE VISUALIZATION SUMMARY")
    print("=" * 70)

    print(f"\nImages evaluated: {load_stats['num_images_evaluated']}")
    print(f"  Skipped (GT only, no prediction file):     {load_stats['num_images_gt_only_skipped']}")
    print(f"  Skipped (prediction only, no GT file):     {load_stats['num_images_pred_only_skipped']}")
    print(f"  Skipped (no matching image found):         {load_stats['num_images_missing_image_skipped']}")
    print(f"  Degraded (no metadata.json entry):         {load_stats['num_images_missing_metadata']}")

    mode = "class-aware" if report["class_aware"] else "class-agnostic"
    print(f"\nMatching: IoU >= {report['iou_threshold']:.2f}, {mode}")
    print(
        f"Aggregate over all evaluated images: "
        f"TP={report['total_tp']} FP={report['total_fp']} FN={report['total_fn']}"
    )
    print(
        f"\nImages with at least one FP or FN: "
        f"{report['num_images_with_errors']} / {report['num_images_evaluated']}"
    )
    print(f"Figures written: {num_figures} → {save_dir}")

    top = report["images"][:10]
    if top:
        print("\n--- Worst images (by FP + FN) ---")
        header = f"{'image':<40}{'TP':>6}{'FP':>6}{'FN':>6}"
        print(header)
        print("-" * len(header))
        for entry in top:
            print(
                f"{entry['stem']:<40}{entry['num_tp']:>6}"
                f"{entry['num_fp']:>6}{entry['num_fn']:>6}"
            )
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find every image where NAUTILUS produced a false positive or a false "
            "negative and render a 2x2 GT / TP / FP / FN comparison for each."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gt-dir", required=True,
        help="Directory of YOLO-format ground-truth label files (<image_stem>.txt).",
    )
    parser.add_argument(
        "--pred-dir", required=True,
        help="Directory of raw NAUTILUS result .txt files, e.g. <batch_inference "
             "output-dir>/results. metadata.json is read from <pred-dir>/../metadata.json.",
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--classes-file", required=True,
        help="File with one class name per line (line index = YOLO class id). Required: "
             "without it GT labels are numeric ids and cannot match NAUTILUS's free-text "
             "labels, making class-aware matching meaningless.",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory for the 2x2 figures (default: <pred-dir>/../error_visualizations).",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="Path for the JSON error report (default: <save-dir>/errors.json).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU required for a prediction to count as a true positive (default: 0.5).",
    )
    parser.add_argument(
        "--class-agnostic", action="store_true",
        help="Ignore labels when matching. Default is class-aware (label must match too).",
    )
    parser.add_argument(
        "--fishies", action="store_true", default=False,
        help="Treat 'fish' and 'small_fish' as a single class (merges 'small_fish' into 'fish').",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search --image-dir recursively.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only write figures for the N worst images (most FP+FN). The JSON report "
             "and printed totals always cover every image.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    metadata_path = pred_dir.parent / "metadata.json"

    for label, p in [("Ground-truth", gt_dir), ("Prediction", pred_dir), ("Image", image_dir)]:
        if not p.exists():
            print(f"[error] {label} directory not found: {p}", file=sys.stderr)
            sys.exit(1)
    if not metadata_path.exists():
        print(
            f"[error] metadata.json not found: {metadata_path} "
            f"(expected as a sibling of --pred-dir, i.e. <pred-dir>/../metadata.json)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_dir else pred_dir.parent / "error_visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json = Path(args.save_json) if args.save_json else save_dir / "errors.json"

    class_names = [
        line.strip() for line in Path(args.classes_file).read_text().splitlines() if line.strip()
    ]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    image_dims = metadata.get("image_dims", {})
    if not image_dims:
        print(f"[warning] metadata.json at {metadata_path} has no 'image_dims' entries", file=sys.stderr)

    dataset, load_stats = load_dataset(
        gt_dir, pred_dir, image_dir, image_dims, class_names, args.recursive, args.fishies
    )

    if not dataset:
        print("[error] No images could be loaded — nothing to visualize.", file=sys.stderr)
        sys.exit(1)

    class_aware = not args.class_agnostic

    # ── Pass 1: split every image, collect the ones with errors ──────────────
    total_tp = total_fp = total_fn = 0
    error_entries = []
    for stem in sorted(dataset):
        record = dataset[stem]
        tp, fp, fn = split_errors(record, args.iou_threshold, class_aware)
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        if not fp and not fn:
            continue

        error_entries.append(
            {
                "stem": stem,
                "num_tp": len(tp),
                "num_fp": len(fp),
                "num_fn": len(fn),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                # Kept out of the JSON report, needed for drawing.
                "_gt": record["gt"],
                "_no_pred_parsed": len(record["pred"]) == 0,
            }
        )

    error_entries.sort(key=lambda e: (e["num_fp"] + e["num_fn"], e["stem"]), reverse=True)

    # ── Pass 2: draw the figures ────────────────────────────────────────────
    to_draw = error_entries if args.limit is None else error_entries[: args.limit]
    print(f"\nRendering {len(to_draw)} error figure(s) to {save_dir}")

    num_figures = 0
    for entry in to_draw:
        stem = entry["stem"]
        image_path = find_image_for_stem(image_dir, stem, args.recursive)
        if image_path is None:
            # load_dataset already resolved this once, so this should not happen.
            print(f"[warning] {stem}: image disappeared between passes — skipping figure", file=sys.stderr)
            continue

        out_path = save_dir / f"{stem}_errors.png"
        draw_error_figure(
            image_path,
            entry["_gt"],
            entry["true_positives"],
            entry["false_positives"],
            entry["false_negatives"],
            out_path,
            stem=stem,
            iou_threshold=args.iou_threshold,
            class_aware=class_aware,
            no_pred_parsed=entry["_no_pred_parsed"],
        )
        entry["image"] = str(image_path.resolve())
        entry["figure"] = str(out_path.resolve())
        num_figures += 1

    # ── Report ──────────────────────────────────────────────────────────────
    for entry in error_entries:
        entry.pop("_gt", None)
        entry.pop("_no_pred_parsed", None)

    report = {
        "iou_threshold": args.iou_threshold,
        "class_aware": class_aware,
        "fishies": args.fishies,
        "num_images_evaluated": load_stats["num_images_evaluated"],
        "num_images_with_errors": len(error_entries),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        **load_stats,
        "images": error_entries,
    }

    print_summary(report, load_stats, num_figures, save_dir)

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nError report written to {save_json}")


if __name__ == "__main__":
    main()
