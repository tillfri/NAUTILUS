import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_yolo_bbox_distributions(base_dir, dataset_names=("SAR", "AB")):
    """
    Creates a 3x3 violin plot grid comparing YOLO bounding box stats from two datasets.

    base_dir: str
        Directory containing the two datasets (each with 'images' and 'labels' subdirs).
    dataset_names: tuple
        Tuple of dataset directory names inside base_dir (e.g., ('SAR', 'AB')).
    """

    def parse_yolo_labels(label_dir):
        widths, heights, areas = [], [], []
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(label_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # skip malformed lines
                    _, _, _, w, h = map(float, parts)
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
        return widths, heights, areas

    data = {"Width": [], "Height": [], "Area": [], "Dataset": []}

    for name in dataset_names:
        label_path = os.path.join(base_dir, name, "labels")
        widths, heights, areas = parse_yolo_labels(label_path)

        data["Width"].extend(widths)
        data["Height"].extend(heights)
        data["Area"].extend(areas)
        data["Dataset"].extend([name] * len(widths))

    df = pd.DataFrame(data)

    # Setup the 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    metrics = ["Width", "Height", "Area"]

    for i, metric in enumerate(metrics):
        # Single dataset plots
        for j, dataset in enumerate(dataset_names):
            sns.violinplot(
                y=df[df["Dataset"] == dataset][metric], ax=axes[i, j], color="skyblue"
            )
            axes[i, j].set_title(f"{metric} - Dataset {dataset}")
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel(metric)

        # Combined overlay plot
        sns.violinplot(
            data=df, x="Dataset", y=metric, ax=axes[i, 2], palette="muted", split=False
        )
        axes[i, 2].set_title(f"{metric} - Both Datasets")
        axes[i, 2].set_xlabel("")
        axes[i, 2].set_ylabel(metric)

    plt.suptitle("Bounding Box Size Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("Violin Plots BoundingBox SAR vs AB", dpi=300)
    plt.close()


def plot_extended_yolo_bbox_distributions(
    base_dir_1,
    base_dir_2,
    dataset_names_1=("SAR", "AB"),
    dataset_names_2=("AB_SAR", "AB_only"),
):
    """
    Creates a 3x5 violin plot grid comparing YOLO bounding box stats from four datasets.

    base_dir_1: str
        First directory containing 2 datasets.
    base_dir_2: str
        Second directory containing 2 more datasets.
    dataset_names_1: tuple
        Names of datasets in base_dir_1.
    dataset_names_2: tuple
        Names of datasets in base_dir_2.
    """

    def parse_yolo_labels(label_dir):
        widths, heights, areas = [], [], []
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(label_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, _, _, w, h = map(float, parts)
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
        return widths, heights, areas

    data = {"Width": [], "Height": [], "Area": [], "Dataset": []}

    all_datasets = []

    # Process datasets from base_dir_1
    for name in dataset_names_1:
        label_path = os.path.join(base_dir_1, name, "labels")
        widths, heights, areas = parse_yolo_labels(label_path)
        data["Width"].extend(widths)
        data["Height"].extend(heights)
        data["Area"].extend(areas)
        data["Dataset"].extend([name] * len(widths))
        all_datasets.append(name)

    # Process datasets from base_dir_2
    for name in dataset_names_2:
        label_path = os.path.join(base_dir_2, name, "labels")
        widths, heights, areas = parse_yolo_labels(label_path)
        data["Width"].extend(widths)
        data["Height"].extend(heights)
        data["Area"].extend(areas)
        data["Dataset"].extend([name] * len(widths))
        all_datasets.append(name)

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    metrics = ["Width", "Height", "Area"]

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(all_datasets):
            sns.violinplot(
                y=df[df["Dataset"] == dataset][metric], ax=axes[i, j], color="skyblue"
            )
            axes[i, j].set_title(f"{metric} - {dataset}")
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel(metric)

        # Final column: Combined plot for all datasets
        sns.violinplot(
            data=df, x="Dataset", y=metric, ax=axes[i, 4], palette="muted", split=False
        )
        axes[i, 4].set_title(f"{metric} - All Datasets")
        axes[i, 4].set_xlabel("")
        axes[i, 4].set_ylabel(metric)

    plt.suptitle("Bounding Box Size Distributions Across 4 Datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("Extended Violin Plots BoundingBox SAR vs AB", dpi=300)
    plt.close()


if __name__ == "__main__":
    # plot_yolo_bbox_distributions("stats_SAR_vs_AB")
    plot_extended_yolo_bbox_distributions("stats_SAR_vs_AB", "stats_SAR_vs_AB/AB_split")
