"""
resolution_histogram.py — Histogram of image resolutions across the nautdata
collection.

nautdata is a curated collection of underwater-image datasets: each immediate
subdirectory of --dir is one source dataset. This script walks every subdir,
recursively finds all images, reads each image's (width, height) from its
header only (no pixel decode), and plots:

  1. A facet grid with one resolution histogram per source dataset.
  2. A single combined histogram for the whole collection, stacked by source
     dataset so each bar still shows which datasets contributed to it.

Resolution is reduced to one scalar — megapixels (width * height / 1e6) — and
binned on a log scale, since resolutions in this collection range from tiny
crops (~0.06 MP) to full HD+ (~2-8 MP).

Because a full scan is slow even with threading (a few minutes for ~450k
images on the nautdata NFS mount), results are cached to a CSV next to the
output plots; re-running with the same --cache re-uses it instantly. Pass
--force-rescan to ignore an existing cache.

Usage:
    python resolution_histogram.py --dir /path/to/nautdata_images
    python resolution_histogram.py --dir /path/to/nautdata_images \\
        --cache resolutions.csv --workers 32 --top-n 7

Arguments:
    --dir             Root directory containing one subdir per dataset.
    --cache           CSV cache path for scanned resolutions
                       (default: resolution_cache.csv).
    --force-rescan    Ignore an existing cache and rescan all images.
    --workers         Thread pool size for image header reads (default: 32).
    --bins            Number of log-spaced bins (default: 40).
    --top-n           Max number of datasets shown individually in the
                       combined chart's stacking/legend; the rest are folded
                       into "Other" (default: 7).
    --facet-output    Output path for the per-dataset facet grid
                       (default: resolution_histogram_by_dataset.png).
    --combined-output Output path for the combined histogram
                       (default: resolution_histogram_combined.png).
    --show            Display the plots interactively in addition to saving.
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Categorical palette (validated order, dataviz skill reference palette).
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
OTHER_COLOR = "#898781"  # muted ink — folded tail is not a categorical entity
MEAN_COLOR = "#DC2626"
MEDIAN_COLOR = "#374151"
GRID_COLOR = "#E5E7EB"


def find_images(dataset_dir: Path) -> list:
    return [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def read_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (width, height); header-only read, no decode
    except Exception:
        return None


def scan_dataset(dataset_dir: Path, workers: int, progress_label: str):
    images = find_images(dataset_dir)
    rows = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, size in zip(images, ex.map(read_size, images)):
            if size is None:
                failed += 1
                continue
            rows.append((dataset_dir.name, size[0], size[1], str(path)))
    print(f"[scan] {progress_label}: {len(rows)} images ok, {failed} failed "
          f"(of {len(images)} found)", file=sys.stderr)
    return rows


def scan_all(root: Path, workers: int) -> list:
    dataset_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dataset_dirs:
        print(f"[error] No subdirectories found in {root}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for i, d in enumerate(dataset_dirs, 1):
        all_rows.extend(scan_dataset(d, workers, f"[{i}/{len(dataset_dirs)}] {d.name}"))
    return all_rows


def load_cache(cache_path: Path) -> list:
    rows = []
    with cache_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for dataset, w, h, path in reader:
            rows.append((dataset, int(w), int(h), path))
    return rows


def save_cache(cache_path: Path, rows: list):
    with cache_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "width", "height", "path"])
        writer.writerows(rows)


def group_top_n(datasets: list, counts: dict, top_n: int):
    """Return (kept_datasets_in_count_order, other_datasets)."""
    ordered = sorted(datasets, key=lambda d: counts[d], reverse=True)
    return ordered[:top_n], ordered[top_n:]


def plot_facets(rows_by_dataset: dict, color_by_dataset: dict, bin_edges: np.ndarray,
                 output: Path, show: bool):
    datasets = sorted(rows_by_dataset, key=lambda d: len(rows_by_dataset[d]), reverse=True)
    n = len(datasets)
    ncols = min(5, n)
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)

    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        mp = rows_by_dataset[dataset]
        ax.hist(mp, bins=bin_edges, color=color_by_dataset[dataset], edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_title(f"{dataset} (n={len(mp)})", fontsize=10)
        ax.set_xlabel("Megapixels", fontsize=8)
        ax.set_ylabel("Images", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Image resolution distribution per nautdata source dataset", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150)
    print(f"Saved per-dataset facet grid to {output}")
    if show:
        plt.show()


def plot_combined(rows_by_dataset: dict, color_by_dataset: dict, other_members: list,
                   bin_edges: np.ndarray, output: Path, show: bool):
    all_mp = np.concatenate(list(rows_by_dataset.values()))
    mean_val = all_mp.mean()
    median_val = np.median(all_mp)

    # Stack order: largest dataset at the bottom, "Other" on top.
    stack_order = sorted(rows_by_dataset, key=lambda d: len(rows_by_dataset[d]), reverse=True)

    widths = np.diff(bin_edges)
    lefts = bin_edges[:-1]
    bottom = np.zeros(len(lefts))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for dataset in stack_order:
        counts, _ = np.histogram(rows_by_dataset[dataset], bins=bin_edges)
        label = dataset if dataset != "Other" else f"Other ({', '.join(other_members)})"
        ax.bar(lefts, counts, width=widths, bottom=bottom, align="edge",
               color=color_by_dataset[dataset], edgecolor="white", linewidth=0.2, label=label)
        bottom += counts

    ax.set_xscale("log")
    ax.axvline(mean_val, color=MEAN_COLOR, linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:.2f} MP")
    ax.axvline(median_val, color=MEDIAN_COLOR, linestyle=":", linewidth=1.5,
               label=f"median = {median_val:.2f} MP")

    ax.set_xlabel("Megapixels (log scale)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"nautdata combined image resolution distribution (n={len(all_mp)} images)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved combined histogram to {output}")
    print(f"n={len(all_mp)} images | mean={mean_val:.2f} MP | median={median_val:.2f} MP "
          f"| min={all_mp.min():.3f} MP | max={all_mp.max():.2f} MP")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-dataset and combined image resolution histograms for nautdata.",
    )
    parser.add_argument("--dir", required=True, help="Root directory containing one subdir per dataset.")
    parser.add_argument("--cache", default="resolution_cache.csv",
                         help="CSV cache path for scanned resolutions (default: resolution_cache.csv).")
    parser.add_argument("--force-rescan", action="store_true",
                         help="Ignore an existing cache and rescan all images.")
    parser.add_argument("--workers", type=int, default=32,
                         help="Thread pool size for image header reads (default: 32).")
    parser.add_argument("--bins", type=int, default=40, help="Number of log-spaced bins (default: 40).")
    parser.add_argument("--top-n", type=int, default=7,
                         help="Max datasets shown individually in the combined chart; "
                              "rest folded into 'Other' (default: 7).")
    parser.add_argument("--facet-output", default="resolution_histogram_by_dataset.png",
                         help="Output path for the per-dataset facet grid.")
    parser.add_argument("--combined-output", default="resolution_histogram_combined.png",
                         help="Output path for the combined histogram.")
    parser.add_argument("--show", action="store_true",
                         help="Display the plots interactively in addition to saving.")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"[error] {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.force_rescan:
        print(f"[cache] Loading cached resolutions from {cache_path}", file=sys.stderr)
        rows = load_cache(cache_path)
    else:
        rows = scan_all(root, args.workers)
        save_cache(cache_path, rows)
        print(f"[cache] Saved {len(rows)} rows to {cache_path}", file=sys.stderr)

    if not rows:
        print("[error] No images found/readable.", file=sys.stderr)
        sys.exit(1)

    rows_by_dataset_raw = {}
    for dataset, w, h, _path in rows:
        rows_by_dataset_raw.setdefault(dataset, []).append(w * h / 1e6)
    rows_by_dataset = {d: np.array(mp) for d, mp in rows_by_dataset_raw.items()}

    counts = {d: len(mp) for d, mp in rows_by_dataset.items()}
    kept, other = group_top_n(list(rows_by_dataset), counts, args.top_n)

    color_by_dataset = {d: CATEGORICAL_COLORS[i] for i, d in enumerate(kept)}

    # Facet grid keeps every dataset separate (small multiples), but reuses
    # the combined chart's colors so a panel can be matched back to its
    # stacked segment; folded-tail datasets get the "Other" gray.
    facet_colors = dict(color_by_dataset)
    for d in other:
        facet_colors[d] = OTHER_COLOR

    all_mp = np.concatenate(list(rows_by_dataset.values()))
    bin_edges = np.geomspace(all_mp.min(), all_mp.max(), args.bins + 1)

    plot_facets(rows_by_dataset, facet_colors, bin_edges, Path(args.facet_output), args.show)

    combined_by_group = {d: rows_by_dataset[d] for d in kept}
    if other:
        combined_by_group["Other"] = np.concatenate([rows_by_dataset[d] for d in other])
        color_by_dataset["Other"] = OTHER_COLOR

    plot_combined(combined_by_group, color_by_dataset, other, bin_edges,
                  Path(args.combined_output), args.show)

    print("\nPer-dataset image counts:")
    for d in sorted(counts, key=counts.get, reverse=True):
        print(f"  {d}: {counts[d]}")


if __name__ == "__main__":
    main()
