"""
visualize_detections.py — Draw bounding boxes from NAUTILUS model output onto an image.

The NAUTILUS inference script rescales bbox coordinates in the prompt from original image
space to model input space (patch-grid resolution) before feeding the model.  The model
therefore outputs bboxes in model input space too.  Pass --input-width / --input-height
(= grid_thw[2]*14 and grid_thw[1]*14 from the inference script) to apply the inverse
scale and map detections back to original image space before drawing.  Omit them only if
your bboxes are already in original image space.

Usage:
    # Bboxes already in original image space (no rescaling needed):
    python visualize_detections.py --image path/to/image.jpg \
        --detections '{"bbox_2d": [363, 144, 833, 650], "label": "jellyfish"}'

    # Bboxes in model input space → rescale to original image space:
    python visualize_detections.py --image path/to/image.jpg \
        --input-width 1176 --input-height 896 \
        --detections '[{"bbox_2d": [363, 144, 833, 650], "label": "jellyfish"}]'

    # From a JSON file:
    python visualize_detections.py --image path/to/image.jpg \
        --input-width 1176 --input-height 896 \
        --detections-file predictions.json

    # From raw model output text (bbox dicts extracted automatically):
    python visualize_detections.py --image path/to/image.jpg \
        --input-width 1176 --input-height 896 \
        --model-output 'A jellyfish {"bbox_2d": [363, 144, 833, 650], "label": "jellyfish"}'

How to get --input-width / --input-height from the inference script:
    grid_thw, ori_w, ori_h = get_grid_thw(image_processor, image_path)
    input_height = grid_thw[1].item() * 14   # → --input-height
    input_width  = grid_thw[2].item() * 14   # → --input-width

Arguments:
    --image             Path to the source image (required).
    --detections        JSON string — a single detection dict or a list of dicts.
    --detections-file   Path to a JSON file containing detections (single dict or list).
    --model-output      Raw text output from the model; all JSON-like bbox dicts are
                        extracted automatically.
    --input-width       Model input width in pixels (grid_thw[2] * 14).  When provided
                        together with --input-height the bboxes are rescaled from model
                        input space to original image space before drawing.
    --input-height      Model input height in pixels (grid_thw[1] * 14).
    --output            Output image path (default: <image_stem>_detections.<ext>).
    --no-display        Skip displaying the image (useful for headless environments).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

# Use a non-interactive backend by default (safe for headless environments)
matplotlib.use("Agg")


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


def rescale_bbox(bbox: list[int], inv_scale_w: float, inv_scale_h: float) -> list[int]:
    """Map a bbox from model input space back to original image space."""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * inv_scale_w),
        round(y1 * inv_scale_h),
        round(x2 * inv_scale_w),
        round(y2 * inv_scale_h),
    ]


def extract_detections_from_text(text: str) -> list[dict]:
    """Pull every JSON object that contains 'bbox_2d' out of free-form model output."""
    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}'
    matches = re.findall(pattern, text)
    detections = []
    for m in matches:
        try:
            detections.append(json.loads(m))
        except json.JSONDecodeError:
            print(f"[warning] Could not parse detection: {m}", file=sys.stderr)
    return detections


def parse_detections(args) -> list[dict]:
    if args.detections:
        text = args.detections.strip()
        # 1. Try valid JSON as-is (a list or a single object)
        try:
            raw = json.loads(text)
            return raw if isinstance(raw, list) else [raw]
        except json.JSONDecodeError:
            pass
        # 2. Try wrapping in [] to handle comma-separated objects
        try:
            raw = json.loads(f"[{text}]")
            return raw if isinstance(raw, list) else [raw]
        except json.JSONDecodeError:
            pass
        # 3. Fall back to regex extraction (same as --model-output)
        detections = extract_detections_from_text(text)
        if detections:
            return detections
        print("[error] Could not parse --detections value as JSON.", file=sys.stderr)
        sys.exit(1)

    if args.detections_file:
        with open(args.detections_file) as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else [raw]

    if args.model_output:
        return extract_detections_from_text(args.model_output)

    print("[error] Provide --detections, --detections-file, or --model-output.", file=sys.stderr)
    sys.exit(1)


def draw(
    image_path: str,
    detections: list[dict],
    output_path: str,
    display: bool,
    inv_scale_w: float | None,
    inv_scale_h: float | None,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    rescaling = inv_scale_w is not None and inv_scale_h is not None
    if rescaling:
        print(
            f"[info] Rescaling bboxes from model input space to original image space "
            f"(inv_scale_w={inv_scale_w:.4f}, inv_scale_h={inv_scale_h:.4f})"
        )

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.axis("off")

    label_map: dict[str, str] = {}
    legend_handles = []
    seen_labels = set()

    for det in detections:
        bbox = det.get("bbox_2d")
        label = det.get("label", "object")

        if bbox is None or len(bbox) != 4:
            print(f"[warning] Skipping detection with missing/invalid bbox: {det}", file=sys.stderr)
            continue

        if rescaling:
            bbox = rescale_bbox(bbox, inv_scale_w, inv_scale_h)

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        colour = get_colour(label, label_map)

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
            label,
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor=colour, edgecolor="none", pad=2, alpha=0.85),
            verticalalignment="bottom",
        )

        if label not in seen_labels:
            legend_handles.append(
                mpatches.Patch(facecolor=colour, edgecolor="white", label=label)
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

    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if display:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NAUTILUS bounding-box detections on an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument(
        "--detections",
        help='JSON string: single dict or list of dicts with "bbox_2d" and "label".',
    )
    parser.add_argument(
        "--detections-file",
        help="Path to a JSON file containing a single detection or a list.",
    )
    parser.add_argument(
        "--model-output",
        help="Raw text output from the model; bbox dicts are extracted automatically.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=None,
        help=(
            "Model input width in pixels (grid_thw[2] * 14 from the inference script). "
            "Required together with --input-height to rescale bboxes from model input "
            "space back to original image space."
        ),
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=None,
        help=(
            "Model input height in pixels (grid_thw[1] * 14 from the inference script). "
            "Required together with --input-width to rescale bboxes from model input "
            "space back to original image space."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output image path (default: <stem>_detections<ext>).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an interactive window (useful for headless environments).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Validate that both or neither rescaling args are provided
    if (args.input_width is None) != (args.input_height is None):
        print("[error] Provide both --input-width and --input-height, or neither.", file=sys.stderr)
        sys.exit(1)

    # Compute inverse scale factors: model input space → original image space
    inv_scale_w = inv_scale_h = None
    if args.input_width is not None:
        ori_w, ori_h = Image.open(image_path).size
        inv_scale_w = ori_w / args.input_width
        inv_scale_h = ori_h / args.input_height

    output_path = args.output or str(
        image_path.parent / f"{image_path.stem}_detections{image_path.suffix}"
    )

    detections = parse_detections(args)
    if not detections:
        print("[warning] No detections found — saving an unmodified copy.", file=sys.stderr)

    draw(
        str(image_path),
        detections,
        output_path,
        display=not args.no_display,
        inv_scale_w=inv_scale_w,
        inv_scale_h=inv_scale_h,
    )


if __name__ == "__main__":
    main()
