#!/usr/bin/env python3
"""
Validate that images from the same video are contained within a single fold
in the 4-fold cross-validation setup.

This script checks that each unique video_id (extracted from filenames as the
number in parentheses) appears in only one validation fold across all dataset
directories.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def extract_video_id(filename: str) -> int | None:
    """
    Extract video_id from filename pattern: ...(NUMBER).mp4...

    Args:
        filename: Label file name containing video_id in parentheses

    Returns:
        Video ID as integer, or None if pattern not found
    """
    match = re.search(r"\((\d+)\)\.mp4", filename)
    if match:
        return int(match.group(1))
    return None


def collect_video_ids_per_fold(base_dir: Path) -> Dict[str, Set[int]]:
    """
    Collect all video IDs present in each fold's validation set.

    Args:
        base_dir: Base directory containing dataset1, dataset2, etc.

    Returns:
        Dictionary mapping fold name to set of video IDs
    """
    fold_video_ids = {}

    # Find all dataset directories
    dataset_dirs = sorted(base_dir.glob("dataset*"))
    dataset_dirs = [d for d in dataset_dirs if d.is_dir()]

    for dataset_dir in dataset_dirs:
        fold_name = dataset_dir.name
        val_labels_dir = dataset_dir / "labels" / "val"

        if not val_labels_dir.exists():
            print(f"Warning: {val_labels_dir} does not exist")
            continue

        video_ids = set()
        label_files = list(val_labels_dir.glob("*.txt"))

        for label_file in label_files:
            video_id = extract_video_id(label_file.name)
            if video_id is not None:
                video_ids.add(video_id)
            else:
                print(f"Warning: Could not extract video_id from {label_file.name}")

        fold_video_ids[fold_name] = video_ids
        print(f"{fold_name}: {len(video_ids)} unique video IDs, {len(label_files)} total files")

    return fold_video_ids


def validate_video_separation(fold_video_ids: Dict[str, Set[int]]) -> bool:
    """
    Validate that each video ID appears in only one fold.

    Args:
        fold_video_ids: Dictionary mapping fold name to set of video IDs

    Returns:
        True if validation passes, False otherwise
    """
    # Track which fold(s) each video_id appears in
    video_to_folds: Dict[int, List[str]] = defaultdict(list)

    for fold_name, video_ids in fold_video_ids.items():
        for video_id in video_ids:
            video_to_folds[video_id].append(fold_name)

    # Find violations
    violations = {video_id: folds for video_id, folds in video_to_folds.items() if len(folds) > 1}

    if violations:
        print("\n❌ VALIDATION FAILED")
        print(f"\nFound {len(violations)} video ID(s) appearing in multiple folds:\n")
        for video_id, folds in sorted(violations.items()):
            print(f"  Video ID {video_id}: appears in {', '.join(sorted(folds))}")
        return False
    else:
        print("\n✅ VALIDATION PASSED")
        print(f"\nAll {len(video_to_folds)} unique video IDs are properly separated across folds.")

        # Print summary statistics
        print("\nFold statistics:")
        for fold_name in sorted(fold_video_ids.keys()):
            count = len(fold_video_ids[fold_name])
            print(f"  {fold_name}: {count} unique videos")

        return True


def main():
    """Main entry point."""
    base_dir = Path("/home/till/projektbericht/datasets/SOR-922-AR/4-fold-cross-val")

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return 1

    print(f"Validating video separation in: {base_dir}\n")
    print("=" * 60)

    # Collect video IDs from all folds
    fold_video_ids = collect_video_ids_per_fold(base_dir)

    if not fold_video_ids:
        print("Error: No dataset directories found")
        return 1

    print("=" * 60)

    # Validate separation
    is_valid = validate_video_separation(fold_video_ids)

    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
