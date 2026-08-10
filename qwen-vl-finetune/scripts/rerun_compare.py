"""
rerun_compare.py — Stage 2 of the NAUTILUS determinism check.

Compares the fresh model outputs collected by rerun_infer.py (in
<run-dir>/error_visualizations/rerun_results/ + rerun_metadata.json) against the
original predictions recorded in <run-dir>/error_visualizations/errors.json
(true_positives + false_positives = the full original prediction set for that image).

For each error image, reports:
  * text_identical  — rerun raw text byte-identical to the original run's
                       results/<stem>.txt
  * boxes_identical — parsed+rescaled boxes are an exact match (same boxes/labels,
                       any order)
  * if not identical, a greedy IoU-based diff (unchanged / added / removed boxes),
    using the same matching logic as evaluate_detections.py.

Usage:
    python rerun_compare.py --run-dir /workspace/runs/<name> [--run-dir ...] \
        [--iou-threshold 0.5] [--output /path/to/report.json]
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

from evaluate_detections import _boxes_and_labels, greedy_match
from visualize_detections import extract_detections_from_text, rescale_bbox


def original_predictions(record: dict) -> list:
    """Reconstruct the full original prediction set for one image from errors.json.

    `true_positives` entries carry extra "iou"/"display" keys that aren't part of a
    raw detection; strip them so the shape matches `false_positives` entries.
    """
    preds = []
    for det in record.get("true_positives", []):
        preds.append({"bbox_2d": det["bbox_2d"], "label": det["label"]})
    for det in record.get("false_positives", []):
        preds.append({"bbox_2d": det["bbox_2d"], "label": det["label"]})
    return preds


def boxes_equal_sets(a: list, b: list) -> bool:
    def normalize(dets):
        return sorted(
            (tuple(round(v) for v in d["bbox_2d"]), str(d["label"]).strip().lower())
            for d in dets
        )

    return normalize(a) == normalize(b)


def diff_predictions(rerun_preds: list, original_preds: list, iou_threshold: float):
    """Greedy IoU-match rerun vs. original predictions; return a breakdown."""
    rerun_boxes, rerun_labels = _boxes_and_labels(rerun_preds)
    orig_boxes, orig_labels = _boxes_and_labels(original_preds)

    matched_pairs, unmatched_rerun, unmatched_orig = greedy_match(
        rerun_boxes, rerun_labels, orig_boxes, orig_labels, iou_threshold, class_aware=True
    )

    unchanged = [
        {"bbox_2d": rerun_boxes[i], "label": rerun_labels[i], "iou": round(iou, 4)}
        for i, _j, iou in matched_pairs
    ]
    added = [
        {"bbox_2d": rerun_boxes[i], "label": rerun_labels[i]}
        for i in sorted(unmatched_rerun)
    ]
    removed = [
        {"bbox_2d": orig_boxes[j], "label": orig_labels[j]}
        for j in sorted(unmatched_orig)
    ]
    return {"unchanged": unchanged, "added": added, "removed": removed}


def compare_run(run_dir: Path, iou_threshold_override, output_path: Path) -> dict:
    errors_path = run_dir / "error_visualizations" / "errors.json"
    rerun_dir = run_dir / "error_visualizations" / "rerun_results"
    rerun_meta_path = run_dir / "error_visualizations" / "rerun_metadata.json"

    with open(errors_path) as f:
        errors = json.load(f)
    rerun_meta = {}
    if rerun_meta_path.exists():
        with open(rerun_meta_path) as f:
            rerun_meta = json.load(f)

    iou_threshold = iou_threshold_override or errors.get("iou_threshold", 0.5)

    results = []
    num_missing = 0
    num_text_identical = 0
    num_boxes_identical = 0

    for record in errors["images"]:
        stem = record["stem"]
        entry = {"stem": stem, "image": record["image"]}

        rerun_text_path = rerun_dir / f"{stem}.txt"
        dims = rerun_meta.get(stem)
        if not rerun_text_path.exists() or dims is None:
            entry["error"] = "no rerun output/metadata found — run rerun_infer.py first"
            results.append(entry)
            num_missing += 1
            continue

        rerun_text = rerun_text_path.read_text()
        original_text_path = run_dir / "results" / f"{stem}.txt"
        original_text = (
            original_text_path.read_text() if original_text_path.exists() else None
        )

        image_path = Path(record["image"])
        ori_w, ori_h = Image.open(image_path).size
        inv_scale_w = ori_w / dims["input_width"]
        inv_scale_h = ori_h / dims["input_height"]

        raw_rerun_dets = extract_detections_from_text(rerun_text)
        rerun_preds = []
        for det in raw_rerun_dets:
            if "label" not in det:
                continue
            rerun_preds.append(
                {
                    "bbox_2d": rescale_bbox(det["bbox_2d"], inv_scale_w, inv_scale_h),
                    "label": str(det["label"]).strip().lower(),
                }
            )

        orig_preds = original_predictions(record)

        text_identical = original_text is not None and rerun_text == original_text
        boxes_identical = boxes_equal_sets(rerun_preds, orig_preds)

        if text_identical:
            num_text_identical += 1
        if boxes_identical:
            num_boxes_identical += 1

        entry.update(
            {
                "text_identical": text_identical,
                "boxes_identical": boxes_identical,
                "original_text": original_text,
                "rerun_text": rerun_text,
                "original_predictions": orig_preds,
                "rerun_predictions": rerun_preds,
            }
        )
        if not boxes_identical:
            entry["diff"] = diff_predictions(rerun_preds, orig_preds, iou_threshold)

        results.append(entry)

    num_evaluated = len(results) - num_missing
    summary = {
        "run_dir": str(run_dir),
        "iou_threshold": iou_threshold,
        "num_images": len(results),
        "num_missing": num_missing,
        "num_evaluated": num_evaluated,
        "num_text_identical": num_text_identical,
        "num_boxes_identical": num_boxes_identical,
        "pct_text_identical": round(100 * num_text_identical / num_evaluated, 2)
        if num_evaluated
        else None,
        "pct_boxes_identical": round(100 * num_boxes_identical / num_evaluated, 2)
        if num_evaluated
        else None,
    }

    report = {"summary": summary, "images": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: compare rerun outputs against original errors.json predictions."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument(
        "--output", type=str, default=None, help="Only valid with a single --run-dir."
    )
    args = parser.parse_args()

    if args.output and len(args.run_dir) > 1:
        parser.error("--output can only be used with a single --run-dir")

    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str)
        output_path = (
            Path(args.output)
            if args.output
            else run_dir / "error_visualizations" / "rerun_check.json"
        )
        print(f"\n=== {run_dir} ===")
        summary = compare_run(run_dir, args.iou_threshold, output_path)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
