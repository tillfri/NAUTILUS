"""
visualize_detections.py — Side-by-side comparison of ground-truth (YOLO format) vs.
NAUTILUS detections, for the output directory produced by batch_inference.py.

batch_inference.py writes:
    <output-dir>/results/<image_stem>.txt   # raw model output text per image
    <output-dir>/metadata.json              # {"image_dims": {"<image_name>": {"input_height":.., "input_width":..}}, ...}

The NAUTILUS inference pipeline rescales bbox coordinates in the prompt from original
image space to model input space (patch-grid resolution) before feeding the model, so
model output bboxes are in model input space too. This script uses the per-image
`input_height`/`input_width` stored in metadata.json to rescale predictions back to
original image space before drawing.

Model output is free-form text and may not contain valid detections at all (e.g. a
plain sentence instead of `{"bbox_2d": [...], "label": "..."}` dicts). Such cases are
handled gracefully: the prediction panel is drawn with no boxes and the raw text is
shown underneath as a note instead of crashing.

Ground-truth annotations are read from a directory of YOLO-format label files (one
`<image_stem>.txt` per image, lines of `class_id cx cy w h` normalized to [0, 1]).
An optional class-names file (one name per line, indexed by class id) can be supplied
to render text labels instead of raw class ids.

Usage:
    python visualize_detections.py \
        --output-dir  /path/to/batch_inference_output \
        --image-dir   /path/to/original/images \
        --gt-dir      /path/to/yolo_labels \
        --classes-file /path/to/classes.txt \
        --limit 20

Arguments:
    --output-dir    Output directory produced by batch_inference.py. Must contain a
                    `results/` subdirectory with per-image `.txt` files, and may
                    contain `metadata.json` (used to rescale predicted bboxes).
    --image-dir     Directory containing the original images that were fed to
                    batch_inference.py (used to look up image dimensions and to
                    render the comparison).
    --gt-dir        Directory containing YOLO-format ground-truth label files
                    (`<image_stem>.txt`, lines `class_id cx cy w h` normalized to
                    [0, 1]). Optional — if omitted, the left panel shows the plain
                    image with no ground-truth boxes.
    --classes-file  Optional text file with one class name per line (line index =
                    YOLO class id). If omitted, raw class ids are used as labels.
    --save-dir      Where to write the side-by-side comparison images (default:
                    `<output-dir>/visualizations`).
    --limit         Max number of images to visualize (default: all available).
    --recursive     Search --image-dir recursively (matches batch_inference.py's
                    --recursive flag).
    --show          Display each comparison interactively (in addition to saving).
                    Off by default since batch runs typically produce many images.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Use a non-interactive backend by default (safe for headless environments)
matplotlib.use("Agg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ── Colour palette (cycles for many labels) ────────────────────────────────────
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def get_colour(label: str, label_map: dict) -> str:
    if label not in label_map:
        label_map[label] = PALETTE[len(label_map) % len(PALETTE)]
    return label_map[label]


def rescale_bbox(bbox: list, inv_scale_w: float, inv_scale_h: float) -> list:
    """Map a bbox from model input space back to original image space."""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * inv_scale_w),
        round(y1 * inv_scale_h),
        round(x2 * inv_scale_w),
        round(y2 * inv_scale_h),
    ]


def extract_detections_from_text(text: str) -> list:
    """Pull every JSON object that contains 'bbox_2d' out of free-form model output.

    Returns an empty list (rather than raising) when the model output does not
    contain any parsable detections — e.g. a plain sentence.
    """
    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}'
    matches = re.findall(pattern, text)
    detections = []
    for m in matches:
        try:
            det = json.loads(m)
        except json.JSONDecodeError:
            continue
        bbox = det.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            det["bbox_2d"] = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        detections.append(det)
    return detections


def load_yolo_gt(
    gt_path: Path, img_w: int, img_h: int, class_names: Optional[list]
) -> list:
    """Parse a YOLO-format label file into [{"bbox_2d": [x1,y1,x2,y2], "label": str}]."""
    if not gt_path.exists():
        return []

    detections = []
    for line_no, line in enumerate(gt_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(
                f"[warning] {gt_path.name}:{line_no}: expected 'class cx cy w h', "
                f"got: {line!r} — skipping",
                file=sys.stderr,
            )
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            print(
                f"[warning] {gt_path.name}:{line_no}: could not parse numbers in "
                f"{line!r} — skipping",
                file=sys.stderr,
            )
            continue

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        if class_names is not None and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = str(cls_id)

        detections.append(
            {"bbox_2d": [round(x1), round(y1), round(x2), round(y2)], "label": label}
        )
    return detections


def find_image_for_stem(image_dir: Path, stem: str, recursive: bool) -> Optional[Path]:
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p
        for p in image_dir.glob(pattern)
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return candidates[0] if candidates else None


def _draw_panel(ax, img_array, detections: list, title: str, label_map: dict) -> None:
    ax.imshow(img_array)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    legend_handles = []
    seen_labels = set()

    for det in detections:
        bbox = det.get("bbox_2d")
        label = det.get("label", "object")

        if bbox is None or len(bbox) != 4:
            print(f"[warning] Skipping detection with missing/invalid bbox: {det}", file=sys.stderr)
            continue

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        colour = get_colour(str(label), label_map)

        rect = mpatches.FancyBboxPatch(
            (x1, y1), w, h,
            boxstyle="square,pad=0",
            linewidth=2,
            edgecolor=colour,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 - 4,
            str(label),
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor=colour, edgecolor="none", pad=2, alpha=0.85),
            verticalalignment="bottom",
        )

        if label not in seen_labels:
            legend_handles.append(
                mpatches.Patch(facecolor=colour, edgecolor="white", label=str(label))
            )
            seen_labels.add(label)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=9,
            framealpha=0.7,
            facecolor="#111111",
            labelcolor="white",
        )


def draw_comparison(
    image_path: Path,
    gt_detections: list,
    pred_detections: list,
    raw_pred_text: str,
    output_path: Path,
    gt_available: bool,
    scale_available: bool,
    show: bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    label_map: dict = {}

    gt_title = "Ground Truth (YOLO)" if gt_available else "Ground Truth (not provided)"
    _draw_panel(axes[0], img_array, gt_detections, gt_title, label_map)

    pred_title = "NAUTILUS Prediction"
    if not scale_available:
        pred_title += "\n(no metadata scale found — bboxes shown as-is)"
    _draw_panel(axes[1], img_array, pred_detections, pred_title, label_map)

    if not pred_detections:
        note = " ".join(raw_pred_text.split())
        if len(note) > 220:
            note = note[:220] + "..."
        axes[1].text(
            0.5, -0.03,
            f"[no bbox parsed from model output] {note!r}",
            transform=axes[1].transAxes,
            ha="center", va="top",
            fontsize=8, color="#cc0000", wrap=True,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


def load_metadata(output_dir: Path) -> dict:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        print(
            f"[warning] No metadata.json found in {output_dir} — predicted bboxes "
            f"will be drawn without rescaling.",
            file=sys.stderr,
        )
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLO ground-truth annotations against NAUTILUS detections for "
            "every image processed by batch_inference.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory produced by batch_inference.py (contains results/ and metadata.json).",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the original images fed to batch_inference.py.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Directory containing YOLO-format ground-truth label files (<image_stem>.txt).",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help="Optional file with one class name per line (line index = YOLO class id).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to write comparison images to (default: <output-dir>/visualizations).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to visualize (default: all available results).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --image-dir recursively (matches batch_inference.py's --recursive flag).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each comparison interactively in addition to saving it.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    if not results_dir.exists():
        print(f"[error] {results_dir} does not exist. Is --output-dir a batch_inference.py output dir?", file=sys.stderr)
        sys.exit(1)

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"[error] Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is not None and not gt_dir.exists():
        print(f"[error] Ground-truth directory not found: {gt_dir}", file=sys.stderr)
        sys.exit(1)

    class_names = None
    if args.classes_file:
        class_names = [
            line.strip()
            for line in Path(args.classes_file).read_text().splitlines()
            if line.strip()
        ]

    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(output_dir)
    image_dims = metadata.get("image_dims", {})

    result_files = sorted(results_dir.glob("*.txt"))
    if not result_files:
        print(f"No result .txt files found in {results_dir}")
        return

    if args.limit is not None:
        result_files = result_files[: args.limit]

    print(f"Visualizing {len(result_files)} result(s) from {results_dir}")

    num_ok = num_missing_image = num_missing_gt = num_missing_scale = 0
    for result_path in result_files:
        stem = result_path.stem
        image_path = find_image_for_stem(image_dir, stem, args.recursive)
        if image_path is None:
            print(f"[warning] No matching image for result {result_path.name} in {image_dir} — skipping", file=sys.stderr)
            num_missing_image += 1
            continue

        raw_text = result_path.read_text()
        pred_detections = extract_detections_from_text(raw_text)

        ori_w, ori_h = Image.open(image_path).size

        dims = image_dims.get(image_path.name)
        if dims is not None:
            inv_scale_w = ori_w / dims["input_width"]
            inv_scale_h = ori_h / dims["input_height"]
            for det in pred_detections:
                det["bbox_2d"] = rescale_bbox(det["bbox_2d"], inv_scale_w, inv_scale_h)
        elif pred_detections:
            num_missing_scale += 1
            print(
                f"[warning] No metadata scale info for {image_path.name} — "
                f"drawing predicted bboxes without rescaling.",
                file=sys.stderr,
            )

        gt_detections = []
        gt_available = False
        if gt_dir is not None:
            gt_path = gt_dir / f"{stem}.txt"
            if gt_path.exists():
                gt_available = True
                gt_detections = load_yolo_gt(gt_path, ori_w, ori_h, class_names)
            else:
                num_missing_gt += 1
                print(f"[warning] No ground-truth label found for {stem} in {gt_dir}", file=sys.stderr)

        out_path = save_dir / f"{stem}_comparison{image_path.suffix}"
        draw_comparison(
            image_path,
            gt_detections,
            pred_detections,
            raw_text,
            out_path,
            gt_available=gt_available,
            scale_available=dims is not None,
            show=args.show,
        )
        num_ok += 1

    print(
        f"Done. Saved: {num_ok}, missing image: {num_missing_image}, "
        f"missing GT: {num_missing_gt}, missing scale info: {num_missing_scale}. "
        f"Comparisons written to {save_dir}"
    )


if __name__ == "__main__":
    main()
