"""
class_annotation_counts.py — Table of annotation counts per class across all
BIIGLE video annotation report CSVs in a directory tree.

Reads every `*.csv` matching `biigle_reports/*/*.csv` (or a custom --glob),
counts rows per distinct `label_name`, and prints a table sorted by count
descending. Optionally writes the table to a CSV file.

Usage:
    python class_annotation_counts.py
    python class_annotation_counts.py --dir biigle_reports --glob "*/*.csv"
    python class_annotation_counts.py --output class_counts.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def count_annotations(root: Path, glob: str) -> Counter:
    csv_paths = sorted(root.glob(glob))
    if not csv_paths:
        print(f"[error] No CSV files found in {root} matching '{glob}'", file=sys.stderr)
        sys.exit(1)

    counts = Counter()
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "label_name" not in (reader.fieldnames or []):
                print(f"[warn] Skipping {csv_path}: no 'label_name' column", file=sys.stderr)
                continue
            for row in reader:
                counts[row["label_name"]] += 1

    print(f"Read {len(csv_paths)} CSV files from {root}", file=sys.stderr)
    return counts


def print_table(counts: Counter):
    total = sum(counts.values())
    name_width = max((len(name) for name in counts), default=5)
    name_width = max(name_width, len("class"))

    header = f"{'class':<{name_width}}  count"
    print(header)
    print("-" * len(header))
    for name, count in counts.most_common():
        print(f"{name:<{name_width}}  {count}")
    print("-" * len(header))
    print(f"{'TOTAL':<{name_width}}  {total}")
    print(f"\n{len(counts)} distinct classes, {total} annotations total")


def write_csv(counts: Counter, output: Path):
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_name", "count"])
        for name, count in counts.most_common():
            writer.writerow([name, count])
    print(f"Saved table to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Count annotations per class across BIIGLE video "
                    "annotation report CSVs.",
    )
    parser.add_argument("--dir", default="biigle_reports",
                        help="Root directory to search (default: biigle_reports).")
    parser.add_argument("--glob", default="*/*.csv",
                        help="Glob pattern relative to --dir (default: '*/*.csv').")
    parser.add_argument("--output", default=None,
                        help="Optional path to also save the table as a CSV file.")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"[error] {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    counts = count_annotations(root, args.glob)
    print_table(counts)

    if args.output:
        write_csv(counts, Path(args.output))


if __name__ == "__main__":
    main()
