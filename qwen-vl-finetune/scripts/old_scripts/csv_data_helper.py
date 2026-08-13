import ast
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def count_labels_and_points(dir1, dir2, output_prefix="output"):
    label_counts = defaultdict(int)
    point_counts = defaultdict(int)

    for directory in [dir1, dir2]:
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                try:
                    df = pd.read_csv(filepath)
                    for _, row in df.iterrows():
                        label = row.get("label_name")
                        points_str = row.get("points")
                        if pd.isna(label) or pd.isna(points_str):
                            continue
                        try:
                            points_list = ast.literal_eval(points_str)
                            if isinstance(points_list, list):
                                label_counts[label] += 1
                                point_counts[label] += len(points_list)
                        except (ValueError, SyntaxError):
                            print(f"Skipping malformed points in {filepath}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    sorted_label_counts = dict(
        sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    )
    sorted_point_counts = dict(
        sorted(point_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Save as JSON
    with open(f"{output_prefix}_label_counts.json", "w") as f:
        json.dump(sorted_label_counts, f, indent=2)
    with open(f"{output_prefix}_point_counts.json", "w") as f:
        json.dump(sorted_point_counts, f, indent=2)


def filter_annotations(
    input_dir: str, output_dir: str, cutoff_date: str = "2025-04-04"
):
    """
    Filters annotation rows from CSV files based on a cutoff date and writes filtered CSVs to an output directory.

    Parameters:
    - input_dir (str): Path to the directory containing input CSV files.
    - output_dir (str): Path to the directory where filtered CSV files will be saved.
    - cutoff_date (str): Date string in format "YYYY-MM-DD" to filter rows created on or after this date.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

    for csv_file in input_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, parse_dates=["created_at"])
            filtered_df = df[df["created_at"] >= cutoff_dt]

            if not filtered_df.empty:
                new_filename = output_path / f"{csv_file.stem}_filtered.csv"
                filtered_df.to_csv(new_filename, index=False)
                print(f"Filtered annotations written to {new_filename}")
            else:
                print(f"No new annotations in {csv_file.name}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


if __name__ == "__main__":
    filter_annotations(
        "annotation_round_3/AB_2022_stand_7.6_csv",
        "annotation_round_3/AB_2022_diff_4.4-7.6_csv",
    )
    # count_labels_and_points(
    #     "updated_annotations_SAR_csv_files", "MGF_AB_csv_video_annotation_report"
    # )
