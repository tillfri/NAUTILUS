import json
import os
import random
import re
import shutil


def organize_files_by_video_id(image_dir, label_dir, output_dir):
    """
    Organizes images and labels into subdirectories based on video ID found in the filename.

    Args:
        image_dir (str): Path to the directory containing image files (.jpg).
        label_dir (str): Path to the directory containing label files (.txt).
        output_dir (str): Path to the directory where organized output should be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(".jpg"):
            continue

        base_name = os.path.splitext(filename)[0]
        match = re.search(r"\((\d+)\)", base_name)
        if not match:
            print(f"Skipping '{filename}': No video ID found in name.")
            continue

        video_id = match.group(1)
        video_dir = os.path.join(output_dir, video_id)
        image_out_dir = os.path.join(video_dir, "images")
        label_out_dir = os.path.join(video_dir, "labels")

        # Create directories if they don't exist
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        # Define source and destination paths
        image_src = os.path.join(image_dir, filename)
        label_filename = base_name + ".txt"
        label_src = os.path.join(label_dir, label_filename)

        image_dst = os.path.join(image_out_dir, filename)
        label_dst = os.path.join(label_out_dir, label_filename)

        try:
            shutil.copy2(image_src, image_dst)
        except Exception as e:
            print(f"Failed to copy image '{filename}': {e}")
            continue

        if not os.path.exists(label_src):
            print(
                f"Label file missing for image '{filename}': Expected '{
                    label_filename
                }'"
            )
            continue

        try:
            shutil.copy2(label_src, label_dst)
        except Exception as e:
            print(f"Failed to copy label '{label_filename}': {e}")


def create_four_fold_split_with_video_ids(root_dir, output_dir, seed=42):
    """
    Splits video_id directories into 4 folds with balanced image counts.
    Copies image and label files flatly into fold_X/images and fold_X/labels.

    Args:
        root_dir (str): Path to root directory containing video_id subdirectories.
        output_dir (str): Where to save fold directories.
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Gather all video_id dirs
    video_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    # Collect image counts per video
    video_data = []
    for vid_dir in video_dirs:
        images_dir = os.path.join(vid_dir, "images")
        image_files = [
            f
            for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
        ]
        video_data.append((vid_dir, image_files))

    # Sort by number of images descending
    video_data.sort(key=lambda x: len(x[1]), reverse=True)

    # Assign to folds greedily
    folds = {f"fold_{i}": [] for i in range(4)}
    fold_counts = {f"fold_{i}": 0 for i in range(4)}

    for vid_dir, image_files in video_data:
        target_fold = min(fold_counts, key=fold_counts.get)
        folds[target_fold].append(vid_dir)
        fold_counts[target_fold] += len(image_files)

    # Copy files
    for fold_name, vid_dirs in folds.items():
        fold_img_dir = os.path.join(output_dir, fold_name, "images")
        fold_lbl_dir = os.path.join(output_dir, fold_name, "labels")
        os.makedirs(fold_img_dir, exist_ok=True)
        os.makedirs(fold_lbl_dir, exist_ok=True)

        for vid_dir in vid_dirs:
            img_src = os.path.join(vid_dir, "images")
            lbl_src = os.path.join(vid_dir, "labels")

            for fname in os.listdir(img_src):
                src_file = os.path.join(img_src, fname)
                if os.path.isfile(src_file):
                    dst_file = os.path.join(fold_img_dir, fname)
                    shutil.copy2(src_file, dst_file)

            for fname in os.listdir(lbl_src):
                src_file = os.path.join(lbl_src, fname)
                if os.path.isfile(src_file):
                    dst_file = os.path.join(fold_lbl_dir, fname)
                    shutil.copy2(src_file, dst_file)

    # Print summary
    print("Fold image distribution:")
    for fold_name in folds:
        total_imgs = len(os.listdir(os.path.join(output_dir, fold_name, "images")))
        print(f"{fold_name}: {total_imgs} images, {len(folds[fold_name])} video IDs")

    return folds


def create_x_fold_split_with_video_ids(root_dir, output_dir, num_folds=4, seed=42):
    """
    Splits video_id directories into `num_folds` with balanced image counts.
    Copies image and label files flatly into fold_X/images and fold_X/labels.

    Args:
        root_dir (str): Path to root directory containing video_id subdirectories.
        output_dir (str): Where to save fold directories.
        num_folds (int): Number of folds to create.
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Gather all video_id dirs
    video_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    # Collect image counts per video
    video_data = []
    for vid_dir in video_dirs:
        images_dir = os.path.join(vid_dir, "images")
        image_files = [
            f
            for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
        ]
        video_data.append((vid_dir, image_files))

    # Sort by number of images descending
    video_data.sort(key=lambda x: len(x[1]), reverse=True)

    # Assign to folds greedily
    folds = {f"fold_{i}": [] for i in range(num_folds)}
    fold_counts = {f"fold_{i}": 0 for i in range(num_folds)}

    for vid_dir, image_files in video_data:
        target_fold = min(fold_counts, key=fold_counts.get)
        folds[target_fold].append(vid_dir)
        fold_counts[target_fold] += len(image_files)

    # Copy files
    for fold_name, vid_dirs in folds.items():
        fold_img_dir = os.path.join(output_dir, fold_name, "images")
        fold_lbl_dir = os.path.join(output_dir, fold_name, "labels")
        os.makedirs(fold_img_dir, exist_ok=True)
        os.makedirs(fold_lbl_dir, exist_ok=True)

        for vid_dir in vid_dirs:
            img_src = os.path.join(vid_dir, "images")
            lbl_src = os.path.join(vid_dir, "labels")

            for fname in os.listdir(img_src):
                src_file = os.path.join(img_src, fname)
                if os.path.isfile(src_file):
                    dst_file = os.path.join(fold_img_dir, fname)
                    shutil.copy2(src_file, dst_file)

            for fname in os.listdir(lbl_src):
                src_file = os.path.join(lbl_src, fname)
                if os.path.isfile(src_file):
                    dst_file = os.path.join(fold_lbl_dir, fname)
                    shutil.copy2(src_file, dst_file)

    # Print summary
    print("Fold image distribution:")
    for fold_name in folds:
        total_imgs = len(os.listdir(os.path.join(output_dir, fold_name, "images")))
        print(f"{fold_name}: {total_imgs} images, {len(folds[fold_name])} video IDs")

    return folds


def create_cross_validation_datasets(folds_dir, output_dir):
    """
    Create 4 cross-validation datasets from 4 folds.
    Each dataset has:
      - 2 folds for training
      - 1 for validation
      - 1 for testing
    Data is copied flat (no subfolders) into:
      datasetX/images/{train,val,test}/
      datasetX/labels/{train,val,test}/
    A config.json is saved per dataset listing which folds were used.

    Args:
        folds_dir (str): Path to fold_{0..3} directories.
        output_dir (str): Where to create dataset1/, dataset2/, ...
    """
    fold_indices = list(range(4))

    # All unique combinations of train (2), val (1), test (1)
    combos = []
    for val_idx in fold_indices:
        for test_idx in fold_indices:
            if test_idx == val_idx:
                continue
            train_idxs = [i for i in fold_indices if i not in (val_idx, test_idx)]
            combos.append((train_idxs, val_idx, test_idx))

    used_combos = set()
    dataset_counter = 1

    for train_idxs, val_idx, test_idx in combos:
        combo_key = tuple(sorted(train_idxs)) + (val_idx, test_idx)
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)

        dataset_name = f"dataset{dataset_counter}"
        dataset_path = os.path.join(output_dir, dataset_name)
        print(
            f"\nCreating {dataset_name} with train: {train_idxs}, val: {
                val_idx
            }, test: {test_idx}"
        )

        # Setup target dirs
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(dataset_path, "images", split), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "labels", split), exist_ok=True)

        # Copy files from folds into appropriate dirs
        for split, idxs in zip(
            ["train", "val", "test"], [train_idxs, [val_idx], [test_idx]]
        ):
            for idx in idxs:
                fold_name = f"fold_{idx}"
                img_src = os.path.join(folds_dir, fold_name, "images")
                lbl_src = os.path.join(folds_dir, fold_name, "labels")

                for fname in os.listdir(img_src):
                    src_file = os.path.join(img_src, fname)
                    dst_file = os.path.join(dataset_path, "images", split, fname)
                    shutil.copy2(src_file, dst_file)

                for fname in os.listdir(lbl_src):
                    src_file = os.path.join(lbl_src, fname)
                    dst_file = os.path.join(dataset_path, "labels", split, fname)
                    shutil.copy2(src_file, dst_file)

        # Write config.json
        config = {
            "train": [f"fold_{i}" for i in train_idxs],
            "val": [f"fold_{val_idx}"],
            "test": [f"fold_{test_idx}"],
        }
        with open(os.path.join(dataset_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        dataset_counter += 1

    print(f"\n✅ Created {dataset_counter - 1} cross-validation datasets.")


def create_cv_datasets_3train_1val(folds_dir, output_dir):
    """
    Create 4 cross-validation datasets from 4 folds.
    Each dataset has:
      - 3 folds for training
      - 1 for validation
    No test set.

    Data is copied flat into:
      datasetX/images/{train,val}/
      datasetX/labels/{train,val}/
    A fold_config.json is saved per dataset listing which folds were used.

    Args:
        folds_dir (str): Path to fold_{0..3} directories.
        output_dir (str): Where to create dataset1/, dataset2/, ...
    """
    fold_indices = list(range(4))
    dataset_counter = 1

    for val_idx in fold_indices:
        train_idxs = [i for i in fold_indices if i != val_idx]
        dataset_name = f"dataset{dataset_counter}"
        dataset_path = os.path.join(output_dir, dataset_name)
        print(f"\nCreating {dataset_name} with train: {train_idxs}, val: {val_idx}")

        # Setup directories
        for split in ["train", "val"]:
            os.makedirs(os.path.join(dataset_path, "images", split), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "labels", split), exist_ok=True)

        # Copy images and labels
        for split, idxs in zip(["train", "val"], [train_idxs, [val_idx]]):
            for idx in idxs:
                fold_name = f"fold_{idx}"
                img_src = os.path.join(folds_dir, fold_name, "images")
                lbl_src = os.path.join(folds_dir, fold_name, "labels")

                for fname in os.listdir(img_src):
                    src_file = os.path.join(img_src, fname)
                    dst_file = os.path.join(dataset_path, "images", split, fname)
                    shutil.copy2(src_file, dst_file)

                for fname in os.listdir(lbl_src):
                    src_file = os.path.join(lbl_src, fname)
                    dst_file = os.path.join(dataset_path, "labels", split, fname)
                    shutil.copy2(src_file, dst_file)

        # Write fold config
        config = {
            "train": [f"fold_{i}" for i in train_idxs],
            "val": [f"fold_{val_idx}"],
        }
        with open(os.path.join(dataset_path, "fold_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        dataset_counter += 1

    print(
        f"\n✅ Created {dataset_counter - 1} 3-train 1-val cross-validation datasets."
    )


if __name__ == "__main__":
    # organize_files_by_video_id(
    #     "round_3_cvat_export/AB_22_organized/images",
    #     "round_3_cvat_export/AB_22_organized/labels",
    #     "images_sorted_by_vid_id",
    # )
    # create_four_fold_split_with_video_ids(
    #     "images_sorted_by_vid_id", "video_folds", seed=42
    # )
    # create_cross_validation_datasets("video_folds", "video_cross_validation")
    # organize_files_by_video_id(
    #     "precise/dataset1/images/val",
    #     "precise/dataset1/labels/val",
    #     "round_1_sorted_by_vid_id",
    # )
    # create_four_fold_split_with_video_ids(
    #     "round_1_sorted_by_vid_id", "folds_round_1_vid_id"
    # )
    # create_cv_datasets_3train_1val(
    #     "folds_round_1_vid_id", "round_1_4_fold_cross_val_id"
    # )
    # create_x_fold_split_with_video_ids(
    #     "images_sorted_by_vid_id", "10_folds_vid_id", num_folds=10
    # )
    # organize_files_by_video_id(
    #     "amrumbank_yolo_dataset/images/val",
    #     "amrumbank_yolo_dataset/labels/val",
    #     "images_sorted_by_vid_id_AFTER_AB",
    # )
    create_x_fold_split_with_video_ids(
        "images_sorted_by_vid_id_AFTER_AB", "video_folds_fix"
    )
