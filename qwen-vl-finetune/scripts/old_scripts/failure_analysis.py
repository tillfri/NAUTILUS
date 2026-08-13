import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def label_folder_to_json(folder_path, image_folder, output_file):
    """
    Converts YOLO-formatted .txt annotation files into a JSON file with bounding boxes in pixel coordinates.

    Parameters:
        folder_path (str): Path to the folder containing YOLO .txt annotation files.
        image_folder (str): Path to the folder containing corresponding JPEG images.
        output_file (str): Path to the output JSON file.
    """
    data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            image_id = os.path.splitext(file_name)[0]  # Remove .txt extension

            image_path = os.path.join(image_folder, image_id + ".jpg")
            if not os.path.exists(image_path):
                print(f"Warning: No matching image found for {file_name}")
                continue

            with Image.open(image_path) as img:
                img_width, img_height = img.size

            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Skip malformed lines

                    category_id = int(parts[0])
                    x_center, y_center, width, height = map(
                        lambda x: round(float(x), 3), parts[1:5]
                    )
                    # Convert from YOLO normalized format to pixel format
                    x_min = round((x_center - width / 2) * img_width, 3)
                    y_min = round((y_center - height / 2) * img_height, 3)
                    bbox_width = round(width * img_width, 3)
                    bbox_height = round(height * img_height, 3)
                    data.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "score": 1.0,  # Assuming a default score of 1.0 (adjust if needed)
                        }
                    )

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=2)

    print(f"JSON saved to {output_file}")


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compare_json_files(groundtruth_file, predictions_file):
    """
    Compare groundtruth and predictions JSON files by matching image_id and computing IoU.

    Parameters:
        groundtruth_file (str): Path to the groundtruth JSON file.
        predictions_file (str): Path to the predictions JSON file.

    Returns:
        dict: Entries from groundtruth with no corresponding match in predictions.
    """
    with open(groundtruth_file, "r") as gt_file:
        groundtruth_data = json.load(gt_file)

    with open(predictions_file, "r") as pred_file:
        predictions_data = json.load(pred_file)

    predictions_dict = {}
    for pred in predictions_data:
        predictions_dict.setdefault(pred["image_id"], []).append(pred)

    unmatched = []

    for gt in groundtruth_data:
        image_id = gt["image_id"]
        category_id = gt["category_id"]
        gt_bbox = gt["bbox"]

        if image_id not in predictions_dict:
            unmatched.append(gt)
            continue

        best_iou = 0
        best_match = None

        for pred in predictions_dict[image_id]:
            if pred["category_id"] == category_id:
                iou = calculate_iou(gt_bbox, pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred

        if best_match:
            predictions_dict[image_id].remove(best_match)
        else:
            unmatched.append(gt)

    return unmatched


def plot_unmatched_bboxes(unmatched, image_folder, output_file):
    """
    Plot all unmatched ground truth bounding boxes in a single image and save to disk.

    Parameters:
        unmatched (list): List of unmatched ground truth entries.
        image_folder (str): Path to the folder containing corresponding images.
        output_file (str): Path to save the final plot.
    """
    num_images = len(unmatched)
    if num_images == 0:
        print("No unmatched images to plot.")
        return

    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7), dpi=150)

    if rows == 1 and cols == 1:
        axes = [[axes]]  # Ensure axes is always 2D
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for idx, entry in enumerate(unmatched):
        image_id = entry["image_id"]
        bbox = entry["bbox"]
        image_path = os.path.join(image_folder, image_id + ".jpg")

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        img = Image.open(image_path)
        ax = axes[idx // cols][idx % cols]
        ax.imshow(img, aspect="auto")
        ax.set_title(image_id)

        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved unmatched bounding boxes plot to {output_file}")


if __name__ == "__main__":
    print(
        len(
            compare_json_files(
                "amrumbank_groundtruth.json",
                "test_amrumbank/test_v8m_640_split_1/predictions.json",
            )
        )
    )
    plot_unmatched_bboxes(
        compare_json_files(
            "amrumbank_groundtruth.json",
            "test_amrumbank/test_v8m_640_split_1/predictions.json",
        ),
        "amrumbank_yolo_dataset/images/val",
        "missed_bboxes_amrumbank",
    )
