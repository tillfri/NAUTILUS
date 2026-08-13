import ast
import json
import os
import random
import re
import shutil

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold


def create_cross_validation_datasets(base_dir, output_base_dir, num_folds=4):
    """
    Split images and labels into a 4-fold cross-validation dataset.

    Args:
        base_dir (str): Path to the base directory containing 'images' and 'labels'.
        output_base_dir (str): Path to create the datasets (e.g., dataset1, dataset2, etc.).
        num_folds (int): Number of folds for cross-validation (default is 4).
    """
    # Define the paths to images and labels
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    # Get all images and corresponding labels
    images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    labels = sorted(
        [f.replace(".jpg", ".txt") for f in images]
    )  # Assume matching label names

    # Ensure all labels exist
    images_and_labels = [
        (img, lbl)
        for img, lbl in zip(images, labels)
        if os.path.exists(os.path.join(labels_dir, lbl))
    ]

    # Shuffle data for randomness
    random.seed(42)  # For reproducibility
    random.shuffle(images_and_labels)

    # Perform K-Fold split
    kf = KFold(n_splits=num_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(images_and_labels), 1):
        # Create directories for this fold
        fold_dir = os.path.join(output_base_dir, f"dataset{fold}")
        os.makedirs(os.path.join(fold_dir, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/val"), exist_ok=True)

        # Get train and validation splits
        train_split = [images_and_labels[i] for i in train_idx]
        val_split = [images_and_labels[i] for i in val_idx]

        # Copy train files
        for img, lbl in train_split:
            shutil.copy(
                os.path.join(images_dir, img),
                os.path.join(fold_dir, "images/train", img),
            )
            shutil.copy(
                os.path.join(labels_dir, lbl),
                os.path.join(fold_dir, "labels/train", lbl),
            )

        # Copy validation files
        for img, lbl in val_split:
            shutil.copy(
                os.path.join(images_dir, img), os.path.join(fold_dir, "images/val", img)
            )
            shutil.copy(
                os.path.join(labels_dir, lbl), os.path.join(fold_dir, "labels/val", lbl)
            )

        print(f"Fold {fold} created in: {fold_dir}")

    print("4-fold cross-validation datasets created successfully!")


def convert_to_yolo_format(bounding_box, frame_width, frame_height):
    """Convert bounding box (x_min, y_min, x_max, y_max) to YOLO format: (x_center, y_center, width, height)."""
    x_min, y_min, x_max, y_max = bounding_box
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    # Normalize to [0, 1] range
    x_center /= frame_width
    y_center /= frame_height
    box_width /= frame_width
    box_height /= frame_height

    return x_center, y_center, box_width, box_height


def is_blurry(frame, threshold=100.0):
    """
    Check if a frame is blurry using the variance of the Laplacian.
    Lower variance indicates more blur.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def is_blurry_canny(frame):
    image_canny = cv2.Canny(frame, 175, 225)
    nonzero_ratio = np.count_nonzero(image_canny) * 1000.0 / image_canny.size
    return nonzero_ratio


def check_dataset_blurry(dataset):
    data_dir = dataset
    img_names = os.listdir(os.path.join(data_dir, "images"))
    blur_stats = []
    for img_name in img_names:
        blur_stats.append(
            is_blurry(cv2.imread(os.path.join(data_dir, "images", img_name)))
        )
    if blur_stats:
        average_blurriness = np.mean(blur_stats)
        print(f"Average Laplacian Variance (Blurriness): {average_blurriness:.2f}")
    plt.figure(figsize=(10, 6))
    plt.hist(blur_stats, bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Laplacian Blurriness", fontsize=16)
    plt.xlabel("Laplacian Variance (Blurriness)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def extract_frames_at_timestamps(
    video_id,
    video_directory,
    timestamps,
    output_folder,
    annotations_list,
    bounding_boxes_list,
    demo=True,
):
    """
    Extracts frames at specified timestamps from a video, optionally saving demo images
    with annotations and YOLO-format bounding box labels.

    Args:
    - video_id (str): Identifier used to match the video file.
    - video_directory (str): Path to the directory containing videos.
    - timestamps (list): Times in seconds to extract frames.
    - output_folder (str): Where to save extracted frames and annotations.
    - annotations_list (list of lists of dict): Per-frame annotations of type 'circle' or 'rectangle'.
    - bounding_boxes_list (list of lists of tuples): Bounding boxes [(x_min, y_min, x_max, y_max)] per frame.
    - demo (bool): If True, saves frames with visual annotation overlays.
    """

    pattern = re.compile(r".*\(({}).*\.mp4".format(re.escape(str(video_id))))
    video_path = ""
    video_name = "NotFound"

    for filename in os.listdir(video_directory):
        if pattern.match(filename):
            video_path = os.path.join(video_directory, filename)
            video_name = filename
            break

    assert video_path != "", f"Video file not found for video_id {video_id}"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    if demo:
        demo_folder = os.path.join(output_folder, "demo")
        os.makedirs(demo_folder, exist_ok=True)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_saved = 0
    frames_failed = 0

    for i, timestamp in enumerate(timestamps):
        frame_number = int(timestamp * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if success:
            frame_filename = (
                f"{output_folder}/{video_name}_at_{timestamp:.2f}_seconds.jpg"
            )
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Extracted frame at {timestamp:.2f} seconds: {frame_filename}")

            annotations = annotations_list[i]
            bounding_boxes = bounding_boxes_list[i]

            # Write YOLO annotations
            yolo_filename = frame_filename.replace(".jpg", ".txt")
            with open(yolo_filename, "w") as f:
                for bbox in bounding_boxes:
                    x_center, y_center, width, height = convert_to_yolo_format(
                        bbox, frame_width, frame_height
                    )
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

            if demo:
                for annotation in annotations:
                    ann_type = annotation.get("type")
                    data = annotation.get("data")

                    if ann_type == "circle" and len(data) == 3:
                        x, y, r = map(int, data)
                        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

                    elif ann_type == "rectangle" and len(data) == 8:
                        pts = [(int(data[i]), int(data[i + 1])) for i in range(0, 8, 2)]
                        for j in range(4):
                            pt1 = pts[j]
                            pt2 = pts[(j + 1) % 4]
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                demo_frame_filename = (
                    f"{demo_folder}/{video_name}_at_{timestamp:.2f}_seconds_demo.jpg"
                )
                cv2.imwrite(demo_frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Saved demo frame with annotations: {demo_frame_filename}")

            frames_saved += 1
        else:
            print(f"Failed to extract frame at {timestamp:.2f} seconds")
            frames_failed += 1

    video.release()
    return frames_saved, frames_failed


def get_time_and_position_from_csv(csv_file, label_id_value):
    """
    Reads a CSV file, filters by the given label_id_value, and extracts annotations.

    Supports both circle ([x, y, r]) and rectangle ([x1, y1, ..., x4, y4]) formats.

    Returns:
    - result_dict (dict): {
        video_id: {
            timestamp: [
                {"type": "circle", "data": [...]},
                {"type": "rectangle", "data": [...]},
                ...
            ],
            ...
        },
        ...
      }
    """
    df = pd.read_csv(csv_file)
    filtered_df = df[df["label_id"] == label_id_value]
    result_dict = {}

    for _, row in filtered_df.iterrows():
        video_id = row["video_id"]
        timestamp = float(row["frames"].strip("[]").split(",")[0])
        annotations = ast.literal_eval(row["points"])

        if video_id not in result_dict:
            result_dict[video_id] = {}
        if timestamp not in result_dict[video_id]:
            result_dict[video_id][timestamp] = []

        for annotation in annotations:
            if len(annotation) == 3:
                annotation_type = "circle"
            elif len(annotation) == 8:
                annotation_type = "rectangle"
            else:
                print(f"Unknown annotation format: {annotation}")
                continue  # skip unknown formats

            entry = {"type": annotation_type, "data": annotation}
            if entry not in result_dict[video_id][timestamp]:
                result_dict[video_id][timestamp].append(entry)

    return result_dict


def annotations_to_bounding_boxes(annotations_list, video_id):
    """
    Converts a list of lists of annotations (each annotation is a dict with 'type' and 'data') to
    a list of bounding boxes per frame.

    - Circles ([x, y, r]) are converted to [x_min, y_min, x_max, y_max]
    - Rectangles ([x1, y1, ..., x4, y4]) are converted to their bounding box.

    Args:
    - annotations_list (list of list of dicts): Each sublist contains multiple annotations for a frame.

    Returns:
    - list of lists of tuples: Each inner list contains bounding boxes for each annotation in a frame.
    """
    bounding_boxes_list = []

    for annotations in annotations_list:
        bounding_boxes = []
        for ann in annotations:
            ann_type = ann.get("type")
            data = ann.get("data")

            if ann_type == "circle" and len(data) == 3:
                x, y, r = data
                bounding_boxes.append((x - r, y - r, x + r, y + r))

            elif ann_type == "rectangle" and len(data) == 8:
                x_coords = data[::2]
                y_coords = data[1::2]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bounding_boxes.append((x_min, y_min, x_max, y_max))

            else:
                print(
                    f"Warning: Unknown or malformed annotation in video '{video_id}': {ann}"
                )
                continue  # Skip malformed annotations

        bounding_boxes_list.append(bounding_boxes)

    return bounding_boxes_list


def process_videos_from_csv(
    csv_file, video_directory, output_folder, label_id_value, demo=False
):
    """
    Reads a CSV file, extracts time and position data (circles and rectangles),
    and processes the videos based on the video_id.
    Calls `extract_frames_at_timestamps` for each video, passing in the timestamps
    and corresponding annotations.

    Args:
    - csv_file (str): Path to the CSV file.
    - video_directory (str): Path to the directory containing video files.
    - output_folder (str): Folder where the extracted frames and annotations will be saved.
    - label_id_value (int): The label_id value to filter by in the CSV file.
    - demo (bool): Flag indicating whether to create demo images with annotations and bounding boxes.
    """
    # Get time and annotations (circle or rectangle) from the CSV file
    data = get_time_and_position_from_csv(csv_file, label_id_value)

    frame_extraction_dict = {}

    for video_id, timestamp_annotations_dict in data.items():
        # Extract the timestamps and the corresponding annotations
        timestamps = list(timestamp_annotations_dict.keys())
        annotations_list = list(timestamp_annotations_dict.values())

        # Convert annotations to bounding boxes
        bounding_boxes_list = annotations_to_bounding_boxes(annotations_list, video_id)
        # Call extract_frames_at_timestamps for the current video
        extracted_frames, failed_frames = extract_frames_at_timestamps(
            video_id=video_id,
            video_directory=video_directory,
            timestamps=timestamps,
            output_folder=output_folder,
            annotations_list=annotations_list,  # Updated to reflect generic handling
            bounding_boxes_list=bounding_boxes_list,
            demo=demo,
        )

        frame_extraction_dict[video_id] = (extracted_frames, failed_frames)

    print("Processing completed for all videos.")
    return frame_extraction_dict


def process_all_annotations(
    output_folder, csv_folder, video_folder_path="MGF_SAR", label_id=252416, demo=False
):
    frame_extraction_dict = {}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for csv_file in os.listdir(csv_folder):
        print(f"Start processing {csv_file}")
        result = process_videos_from_csv(
            os.path.join(csv_folder, csv_file),
            video_folder_path,
            os.path.join(output_folder, csv_file[:-4]),
            label_id,
            demo=demo,
        )
        frame_extraction_dict.update(result)
    print(frame_extraction_dict)
    sum_success = 0
    sum_failed = 0
    for key in frame_extraction_dict.keys():
        sum_success += frame_extraction_dict[key][0]
        sum_failed += frame_extraction_dict[key][1]
    print(f"Extracted frames : {sum_success}")
    print(f"Failed frames : {sum_failed}")


def count_rows_with_column_value(directory, column_name, value):
    """
    Reads all CSV files in a given directory and counts rows where column_name equals the specified value.

    Args:
        directory (str): Path to the directory containing CSV files.
        column_name (str): Name of the column to check.
        value (any): Value to count in the specified column.

    Returns:
        int: Total count of matching rows across all CSV files in the directory.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Provided path '{directory}' is not a directory.")

    total_count = 0
    # Loop through all files in the directory
    for file_name in os.listdir(directory):
        # Process only CSV files
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(file_path)
                # Check if the column exists in the DataFrame
                if column_name in df.columns:
                    # Count rows where the column matches the value
                    count = (df[column_name] == value).sum()
                    print(f"File: {file_name} | Matches: {count}")
                    total_count += count
                else:
                    print(
                        f"File: {file_name} does not contain the column '{column_name}'"
                    )
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
    return total_count


def create_cross_val_from_existing_cross_val(
    existing_crossval_dir, base_dataset_dir, output_dir
):
    """
    Processes subdirectories to copy `.txt` and `.jpg` files based on the directory structure.

    Args:
        existing_crossval_dir (str): The directory containing subdirectories with `labels/val` and `labels/train`.
        base_dataset_dir (str): The base directory containing `images` and `labels` directories.
        output_dir (str): The destination directory for copied files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each subdirectory in input_dir
    for subdirectory in os.listdir(existing_crossval_dir):
        sub_path = os.path.join(existing_crossval_dir, subdirectory)
        if os.path.isdir(sub_path):
            for data_type in ["val", "train"]:
                # Define paths for labels/val and labels/train
                label_path = os.path.join(sub_path, "labels", data_type)
                if os.path.exists(label_path):
                    # Get all .txt files in the val/train folder
                    txt_files = [
                        f for f in os.listdir(label_path) if f.endswith(".txt")
                    ]

                    # Prepare output directory for labels
                    new_label_dir = os.path.join(
                        output_dir, subdirectory, "labels", data_type
                    )
                    os.makedirs(new_label_dir, exist_ok=True)

                    # Copy corresponding .txt files from base_dir to output
                    for txt_file in txt_files:
                        source_file = os.path.join(base_dataset_dir, "labels", txt_file)
                        dest_file = os.path.join(new_label_dir, txt_file)
                        if os.path.exists(source_file):
                            shutil.copy(source_file, dest_file)

                # Define paths for images/val and images/train
                image_path = os.path.join(sub_path, "images", data_type)
                if os.path.exists(image_path):
                    # Get all .jpg files in the val/train folder
                    jpg_files = [
                        f for f in os.listdir(image_path) if f.endswith(".jpg")
                    ]

                    # Prepare output directory for images
                    new_image_dir = os.path.join(
                        output_dir, subdirectory, "images", data_type
                    )
                    os.makedirs(new_image_dir, exist_ok=True)

                    # Copy corresponding .jpg files from base_dir to output
                    for jpg_file in jpg_files:
                        source_file = os.path.join(base_dataset_dir, "images", jpg_file)
                        dest_file = os.path.join(new_image_dir, jpg_file)
                        if os.path.exists(source_file):
                            shutil.copy(source_file, dest_file)


def parse_yolo_label_file(label_file_path):
    """Parses a YOLO format label file and returns the number of instances."""
    with open(label_file_path, "r") as f:
        lines = f.readlines()
    return len(lines)  # Each line corresponds to one instance


def get_dataset_statistics(data_dir):
    """Collects statistics from a YOLO dataset with bounding box sizes organized by file."""
    stats = {
        "train": {"num_images": 0, "num_instances_per_image": [], "bbox_sizes": {}},
        "val": {"num_images": 0, "num_instances_per_image": [], "bbox_sizes": {}},
    }

    for split in ["train", "val"]:
        images_dir = os.path.join(data_dir, "images", split)
        labels_dir = os.path.join(data_dir, "labels", split)

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue

        # Count number of images
        image_files = [
            f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        stats[split]["num_images"] = len(image_files)

        # Count instances per image and collect bounding box sizes
        for image_file in image_files:
            label_file = os.path.join(
                labels_dir, os.path.splitext(image_file)[0] + ".txt"
            )
            file_key = os.path.splitext(image_file)[0]  # File name without extension

            if os.path.exists(label_file):
                bbox_sizes = []
                num_instances = 0

                with open(label_file, "r") as f:
                    lines = f.readlines()
                    num_instances = len(lines)
                    for line in lines:
                        _, x_center, y_center, width, height = map(float, line.split())
                        bbox_sizes.append((width, height))

                # Store the bounding box sizes and instance count
                stats[split]["bbox_sizes"][file_key] = bbox_sizes
                stats[split]["num_instances_per_image"].append(num_instances)
            else:
                stats[split]["bbox_sizes"][file_key] = []
                stats[split]["num_instances_per_image"].append(0)

    return stats


def plot_statistics_for_splits(dataset_path):
    """Plots statistics for all datasets in the given path."""
    dataset_folders = [
        os.path.join(dataset_path, folder)
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
        and folder.startswith("dataset")
    ]

    for dataset in dataset_folders:
        dataset_name = os.path.basename(dataset)
        stats = get_dataset_statistics(dataset)

        # Plot number of images in train and val
        plt.figure(figsize=(10, 5))
        plt.bar(
            ["Train", "Validation"],
            [stats["train"]["num_images"], stats["val"]["num_images"]],
        )
        plt.title(f"Number of Images in Train and Val ({dataset_name})")
        plt.ylabel("Number of Images")
        plt.show()

        # Plot histogram of instances per image for train and val
        for split in ["train", "val"]:
            plt.figure(figsize=(10, 5))
            plt.hist(
                stats[split]["num_instances_per_image"],
                bins=range(
                    0, max(stats[split]["num_instances_per_image"], default=0) + 2
                ),
                edgecolor="black",
            )
            plt.title(
                f"Histogram of Instances per Image ({split.capitalize()} - {dataset_name})"
            )
            plt.xlabel("Number of Instances per Image")
            plt.ylabel("Frequency")
            plt.show()

        # Plot all images in val as a single overview (if required)
        val_images_dir = os.path.join(dataset, "images", "val")
        if os.path.exists(val_images_dir):
            val_image_files = [
                os.path.join(val_images_dir, f)
                for f in os.listdir(val_images_dir)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]

            num_images = len(val_image_files)
            if num_images > 0:
                cols = 5
                rows = (num_images + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
                for i, ax in enumerate(axes.flatten()):
                    if i < num_images:
                        img = plt.imread(val_image_files[i])
                        ax.imshow(img)
                        ax.axis("off")
                    else:
                        ax.remove()
                plt.suptitle(f"All Validation Images Overview ({dataset_name})")
                plt.show()

        # Plot sizes of bounding boxes in train and val
        train_bbox_sizes = [
            bbox
            for file_bboxes in stats["train"]["bbox_sizes"].values()
            for bbox in file_bboxes
        ]
        val_bbox_sizes = [
            bbox
            for file_bboxes in stats["val"]["bbox_sizes"].values()
            for bbox in file_bboxes
        ]
        if train_bbox_sizes or val_bbox_sizes:
            train_widths, train_heights = (
                zip(*train_bbox_sizes) if train_bbox_sizes else ([], [])
            )
            val_widths, val_heights = (
                zip(*val_bbox_sizes) if val_bbox_sizes else ([], [])
            )

            plt.figure(figsize=(10, 5))
            plt.scatter(
                train_widths, train_heights, alpha=0.5, color="red", label="Train"
            )
            plt.scatter(
                val_widths, val_heights, alpha=0.5, color="blue", label="Validation"
            )
            plt.title(
                f"Sizes of Bounding Boxes (Train and Validation - {dataset_name})"
            )
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.legend()
            plt.show()


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union if union > 0 else 0


def process_and_compare(image_dir, label_dir, json_path, output_path):
    # Load inference results
    with open(json_path, "r") as f:
        inference_data = json.load(f)

    results = []

    # Group JSON bounding boxes by image_id
    image_to_json_bboxes = {}
    for entry in inference_data:
        image_id = entry["image_id"]
        if image_id not in image_to_json_bboxes:
            image_to_json_bboxes[image_id] = []
        image_to_json_bboxes[image_id].append(
            {
                "bbox": entry["bbox"],
                "category_id": entry["category_id"],
                "score": entry["score"],
            }
        )

    # Check for label files that are not in image_to_json_bboxes
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for label_file in label_files:
        image_id = os.path.splitext(label_file)[0]
        if image_id not in image_to_json_bboxes:
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}")
                continue

            # Read image dimensions
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # Read and process the label file
            with open(label_path, "r") as f:
                label_bboxes = []
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    label_bboxes.append(
                        {"class_id": class_id, "bbox": [x_min, y_min, width, height]}
                    )

            # Add unmatched label bounding boxes to results
            for label_bbox in label_bboxes:
                results.append(
                    {
                        "image_id": image_id,
                        "label_bbox": label_bbox["bbox"],
                        "iou": 0,
                        "label_only": True,
                    }
                )

    # Process each image_id in image_to_json_bboxes
    for image_id, json_bboxes in image_to_json_bboxes.items():
        # Read image dimensions
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Read corresponding label file
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            label_bboxes = []
        else:
            with open(label_path, "r") as f:
                label_bboxes = []
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    label_bboxes.append(
                        {"class_id": class_id, "bbox": [x_min, y_min, width, height]}
                    )

        # Compare and match bounding boxes
        used_label_indices = set()
        for json_bbox_entry in json_bboxes:
            json_bbox = json_bbox_entry["bbox"]
            score = json_bbox_entry["score"]
            best_iou = 0
            best_match = None
            best_index = None

            for i, label_bbox in enumerate(label_bboxes):
                iou = calculate_iou(json_bbox, label_bbox["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = label_bbox
                    best_index = i

            if best_iou > 0.3:
                results.append(
                    {
                        "image_id": image_id,
                        "json_bbox": json_bbox,
                        "label_bbox": best_match["bbox"],
                        "iou": best_iou,
                        "score": score,
                    }
                )
                used_label_indices.add(best_index)
            else:
                results.append(
                    {
                        "image_id": image_id,
                        "json_bbox": json_bbox,
                        "iou": 0,
                        "json_only": True,
                        "score": score,
                    }
                )

        # Add unmatched label bounding boxes
        for i, label_bbox in enumerate(label_bboxes):
            if i not in used_label_indices:
                results.append(
                    {
                        "image_id": image_id,
                        "label_bbox": label_bbox["bbox"],
                        "iou": 0,
                        "label_only": True,
                    }
                )

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def plot_bbox_with_iou(dataset_path):
    """
    For each dataset in the given path, plots the sizes of bounding boxes in the validation set.
    Points are color-coded by IoU values extracted from resultsSplitX.json.
    Additionally, plots the sizes of training bounding boxes in blue.
    """
    dataset_folders = [
        os.path.join(dataset_path, folder)
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
        and folder.startswith("dataset")
    ]
    miss_counter = 0
    for dataset in dataset_folders:
        dataset_name = os.path.basename(dataset)
        stats = get_dataset_statistics(dataset)

        # Path to resultsSplitX.json
        results_file = f"resultsSplit{dataset[-1]}.json"
        if not os.path.exists(results_file):
            print(f"Results file not found for {dataset_name}. Skipping.")
            continue

        # Load resultsSplitX.json
        with open(results_file, "r") as f:
            results = json.load(f)

        # Validation bounding boxes (from stats dictionary)
        val_bbox_sizes = stats["val"]["bbox_sizes"]

        # Training bounding boxes (from stats dictionary)
        train_bbox_sizes = [
            bbox
            for file_bboxes in stats["train"]["bbox_sizes"].values()
            for bbox in file_bboxes
        ]

        # Initialize lists for validation plotting
        val_widths = []
        val_heights = []
        iou_colors = []  # IoU determines color intensity

        # Iterate through validation bounding boxes
        for image_id, bboxes in val_bbox_sizes.items():
            image_path = os.path.join(dataset, "images", "val", f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"Image {image_id} not found. Skipping.")
                continue

            # Open the image to get its dimensions
            img = plt.imread(image_path)
            img_height, img_width = img.shape[:2]

            for bbox in bboxes:
                # Convert YOLO bbox (normalized) to pixel coordinates
                width, height = bbox
                pixel_bbox = [
                    width * img_width,
                    height * img_height,
                ]
                potential_matches = []
                # Match with results based on image_id and bounding box
                matched_iou = None
                for result in results:
                    if (
                        result["image_id"] == image_id
                        and "label_bbox" in result
                        and "label_only" not in result
                    ):
                        label_bbox = result["label_bbox"]
                        # Compare pixel_bbox with label_bbox
                        if (
                            abs(pixel_bbox[0] - label_bbox[2]) < 1e-3
                            and abs(pixel_bbox[1] - label_bbox[3]) < 1e-3
                        ):
                            matched_iou = result["iou"]
                            break  # Assume a single match for simplicity
                        else:
                            potential_matches.append(label_bbox)
                    elif result["image_id"] == image_id and "label_only" in result:
                        matched_iou = 0
                        miss_counter = miss_counter + 1
                        print(
                            f"Found label_only for {image_id}, added miss {miss_counter}"
                        )

                if matched_iou is not None:
                    val_widths.append(width)
                    val_heights.append(height)
                    iou_colors.append(matched_iou)
                else:
                    print(f"No match found for bbox {pixel_bbox} in {image_id}.")
                    print(f"Potential Matches : {potential_matches}")

        # Prepare training bounding box sizes for plotting
        train_widths = [bbox[0] for bbox in train_bbox_sizes]
        train_heights = [bbox[1] for bbox in train_bbox_sizes]

        # Define a custom colormap for IoU values
        iou_cmap = mcolors.LinearSegmentedColormap.from_list(
            "iou_cmap", [(0, "red"), (0.3, "yellow"), (1, "green")], N=256
        )

        # Plot training bounding box sizes
        plt.figure(figsize=(12, 6), dpi=200)
        plt.scatter(
            train_widths,
            train_heights,
            color="blue",
            alpha=0.5,
            label="Train BBoxes",
        )

        # Plot validation bounding box sizes
        scatter = plt.scatter(
            val_widths,
            val_heights,
            c=iou_colors,
            cmap=iou_cmap,  # Custom color map for IoU
            alpha=0.7,
            label="Validation BBoxes",
        )
        plt.colorbar(scatter, label="IoU Score")
        plt.title(f"Bounding Box Sizes with IoU ({dataset_name})")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # process_all_annotations(
    #     "biigle_extracted_annotations_april",
    #     "updated_annotations_SAR",
    #     video_folder_path="MGF_SAR",
    # )
    process_all_annotations(
        "round_3_cvat_preparation/AB_2022",
        "annotation_round_3/AB_2022_diff_4.4-7.6_csv",
        video_folder_path="MGF_AB",
    )
    # plot_bbox_with_iou("benthos_cross_val")
    # process_and_compare(
    #     image_dir="benthos_cross_val/dataset4/images/val",
    #     label_dir="benthos_cross_val/dataset4/labels/val",
    #     json_path="predictionsSplit4.json",
    #     output_path="resultsSplit4.json",
    # )
    # plot_statistics_for_splits("benthos_cross_val")
    # create_cross_val_from_existing_cross_val('benthos_cross_val', 'dataset_original_annotations', 'cross_val_original_annotations')
    # organize_images_and_labels('original_annotations', 'dataset_test1/images', 'dataset_original_annotations')
    # create_cross_validation_datasets('dataset_original_annotations', 'cross_val_original_annotations', 4)
    # test_dict = get_time_and_position_from_csv("annotations/8887-mgf-sar-hol01-405-sar30-ii.csv", 252416)
    # process_videos_from_csv("annotations/8887-mgf-sar-hol01-405-sar30-ii.csv", "MGF_SAR", "test", 252416, demo=True)
    # process_all_annotations(output_folder="original_annotations")
    # get_time_and_position_from_csv("annotations/8901-mgf-sar-hol01-405-sar30-i.csv", 252416)
    # api_test('1745', "MGF_AB"
