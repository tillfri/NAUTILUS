"""
bbox_histogram.py — Histogram of the number of bboxes per image, given a directory
of per-image `.txt` files in either of two formats:

  * YOLO-format ground-truth labels: one bbox per line, `class_id cx cy w h`.
  * Raw NAUTILUS model output (batch_inference.py's `results/` folder): free-form
    text containing zero or more `{"bbox_2d": [x1,y1,x2,y2], "label": "..."}`
    dicts (comma-separated, possibly with surrounding prose).

The format is auto-detected per file (a file is treated as NAUTILUS output if it
contains the literal string "bbox_2d", otherwise as YOLO), so a single run can't
mix the two per file but the two kinds of directories both just work. An empty
file counts as 0 bboxes.

Usage:
    python bbox_histogram.py --dir /path/to/yolo_labels
    python bbox_histogram.py --dir /path/to/batch_inference_output/results --output hist.png

Arguments:
    --dir          Directory containing per-image `.txt` files (YOLO labels or
                    raw NAUTILUS output).
    --output       Output image path (default: bbox_per_image_histogram.png).
    --recursive    Search --dir recursively for `.txt` files.
    --bins         Number of histogram bins (default: one bin per integer count).
    --show         Display the plot interactively (in addition to saving).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Reuse the already-implemented, well-tested NAUTILUS output parser instead of
# duplicating it.
from visualize_detections import extract_detections_from_text

BAR_COLOR = "#3B82F6"
MEAN_COLOR = "#DC2626"
MEDIAN_COLOR = "#374151"


def count_bboxes_per_image(label_dir: Path, recursive: bool) -> list:
    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = sorted(label_dir.glob(pattern))
    if not txt_files:
        print(f"[error] No .txt files found in {label_dir}", file=sys.stderr)
        sys.exit(1)

    counts = []
    for txt_path in txt_files:
        text = txt_path.read_text()
        if "bbox_2d" in text:
            counts.append(len(extract_detections_from_text(text)))
        else:
            lines = [l for l in text.splitlines() if l.strip()]
            counts.append(len(lines))
    return counts


def plot_histogram(counts: list, output: Path, bins: int, show: bool):
    counts_arr = np.array(counts)
    mean_val = counts_arr.mean()
    median_val = np.median(counts_arr)
    max_val = counts_arr.max()

    if bins is None:
        # One bin per integer bbox count, edges centered on the integers.
        bin_edges = np.arange(-0.5, max_val + 1.5, 1)
    else:
        bin_edges = bins

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(counts_arr, bins=bin_edges, color=BAR_COLOR, edgecolor="white", linewidth=0.5)

    ax.axvline(mean_val, color=MEAN_COLOR, linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:.1f}")
    ax.axvline(median_val, color=MEDIAN_COLOR, linestyle=":", linewidth=1.5,
               label=f"median = {median_val:.0f}")

    ax.set_xlabel("Bboxes per image")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Bbox-per-image distribution (n={len(counts_arr)} images)")
    if max_val <= 30:
        ax.set_xticks(range(0, int(max_val) + 1))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved histogram to {output}")
    print(f"n={len(counts_arr)} images | mean={mean_val:.2f} | median={median_val:.0f} "
          f"| min={counts_arr.min()} | max={max_val}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot a histogram of bboxes-per-image given a directory of "
                    "YOLO-format label .txt files.",
    )
    parser.add_argument("--dir", required=True, help="Directory of YOLO label .txt files.")
    parser.add_argument("--output", default="bbox_per_image_histogram.png",
                        help="Output image path (default: bbox_per_image_histogram.png).")
    parser.add_argument("--recursive", action="store_true",
                        help="Search --dir recursively for .txt files.")
    parser.add_argument("--bins", type=int, default=None,
                        help="Number of histogram bins (default: one bin per integer count).")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively in addition to saving.")
    args = parser.parse_args()

    label_dir = Path(args.dir)
    if not label_dir.is_dir():
        print(f"[error] {label_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    counts = count_bboxes_per_image(label_dir, args.recursive)
    plot_histogram(counts, Path(args.output), args.bins, args.show)


if __name__ == "__main__":
    main()
