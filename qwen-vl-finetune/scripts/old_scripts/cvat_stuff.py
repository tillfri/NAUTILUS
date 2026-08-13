import os
import shutil


def long_version_of_image_names(image_names, label_path, include_jpeg_suffix=True):
    labels = os.listdir(label_path)
    labels = {
        label[3 : 3 + 6]: label if include_jpeg_suffix else label[0:-4]
        for label in labels
    }
    image_names = [
        labels[image_name] if len(image_name) == 6 else image_name
        for image_name in image_names
        if len(image_name) != 6 or image_name in labels
    ]
    return image_names


def organize_images_and_labels(labels_dir, images_dir, output_dir):
    """
    Organize matching labels and images into a specified directory structure.

    Args:
        labels_dir (str): Path to the directory containing label text files.
        images_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory that will be created for organized files.

    The function will create the output directory with two subdirectories:
    - 'images' for the matching images
    - 'labels' for the matching labels
    """
    # Ensure the output directory and subdirectories exist
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Get all label filenames
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    # Process each label file
    for label_file in label_files:
        # Replace '.txt' with '.jpg' to find the matching image
        image_file = label_file.replace(".txt", ".jpg")

        # Check if the matching image exists in the images directory
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(image_path):
            # Copy the label file to the output labels directory
            shutil.copy(label_path, os.path.join(labels_output_dir, label_file))

            # Copy the matching image file to the output images directory
            shutil.copy(image_path, os.path.join(images_output_dir, image_file))

            print(f"Copied: {label_file} and {image_file}")
        else:
            print(f"Image not found for label: {label_file}")

    print(f"Files organized in: {output_dir}")


def organize_files(src_dir, out_dir):
    """
    Organizes .jpg and .txt files from src_dir into subdirectories under out_dir,
    while ignoring the 'demo' subdirectory.

    Args:
        src_dir (str): Source directory to scan for files.
        out_dir (str): Output directory to save files.
    """
    # Create the target subdirectories if they don't exist
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Walk through all subdirectories of the source directory
    for root, dirs, files in os.walk(src_dir):
        # Ignore 'demo' directory
        if "demo" in dirs:
            dirs.remove("demo")  # This prevents os.walk from traversing into 'demo'

        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".jpg"):
                shutil.copy(file_path, os.path.join(images_dir, file))
            elif file.endswith(".txt"):
                shutil.copy(file_path, os.path.join(labels_dir, file))


def convert_dataset_to_cvat(
    codename, label_path, image_names=None, include_images=False
):
    """
    Create a folder with images you want to load into cvat for further annotation and specify that folder in
    path_images. Further, specify a folder with annotations for these images in the yolo format. Note that the
    specified folder is allowed to contain more annotations that are not relevant. Lastly specify a codename for
    your cvat folder. You can then use the .zip file generated in cvat/your_codename/data.zip to load your existing
    annotations into cvat.
    """
    if image_names is None:
        image_names = os.listdir("annotations/images")
    image_names = long_version_of_image_names(
        image_names, label_path, include_jpeg_suffix=False
    )
    to_dir = "./cvat/" + codename
    if os.path.isdir(to_dir):
        shutil.rmtree(to_dir)
    os.mkdir(to_dir)
    project_folder = to_dir
    codename = codename + "/data"
    parent_dir = "./cvat/" + codename
    os.mkdir(parent_dir)
    shutil.copy2("./cvat/obj.data", parent_dir)
    shutil.copy2("./cvat/obj.names", parent_dir)
    with open(parent_dir + "/train.txt", "w") as f:
        for x in range(len(image_names)):
            if x + 1 < len(image_names):
                f.write("obj_train_data/" + image_names[x] + ".jpeg" + "\n")
            else:
                f.write("obj_train_data/" + image_names[x] + ".jpeg")
    os.mkdir(parent_dir + "/obj_train_data")
    for image_name in image_names:
        if image_name.endswith(".jpeg"):
            image_name = image_name[0:-5]
        site = image_name[0:2]
        shutil.copy2(
            os.path.join(label_path, image_name + ".txt"),
            parent_dir + "/obj_train_data",
        )
        if include_images:
            shutil.copy2(
                os.path.join("Mros", "jpeg", site, image_name + ".jpeg"),
                parent_dir + "/obj_train_data",
            )
    shutil.make_archive(os.path.join(project_folder, "data"), "zip", parent_dir)


def convert_dataset_to_cvat_old(label_path, path_images, codename):
    """
    Create a folder with images you want to load into cvat for further annotation and specify that folder in
    path_images. Further, specify a folder with annotations for these images in the yolo format. Note that the
    specified folder is allowed to contain more annotations that are not relevant. Lastly specify a codename for
    your cvat folder. You can then use the .zip file generated in cvat/your_codename/data.zip to load your existing
    annotations into cvat.
    """
    os.mkdir("cvat/" + codename)
    project_folder = "cvat/" + codename
    codename = codename + "/data"
    image_names = os.listdir(path_images)
    parent_dir = "cvat/" + codename
    os.mkdir(parent_dir)
    shutil.copy2("cvat/obj.data", parent_dir)
    shutil.copy2("cvat/obj.names", parent_dir)
    with open(parent_dir + "/train.txt", "w") as f:
        for x in range(len(image_names)):
            if x + 1 < len(image_names):
                f.write("obj_train_data/" + image_names[x] + "\n")
            else:
                f.write("obj_train_data/" + image_names[x])
    labels = os.listdir(label_path)
    labels = list(map(lambda x: x[0:-4], labels))
    os.mkdir(parent_dir + "/obj_train_data")
    for image in image_names:
        if image[-4:] == ".jpg":
            if image[0:-4] in labels:
                shutil.copy2(
                    os.path.join(label_path, image[0:-4] + ".txt"),
                    parent_dir + "/obj_train_data",
                )
                shutil.copy2(
                    os.path.join(path_images, image), parent_dir + "/obj_train_data"
                )
            else:
                open(
                    os.path.join(parent_dir, "obj_train_data", image[0:-5] + ".txt"),
                    "x",
                )
    shutil.make_archive(os.path.join(project_folder, "data"), "zip", parent_dir)


def copy_unmatched_files(folder_annotated, folder_new, output_folder):
    annotated_labels = os.path.join(folder_annotated, "labels")
    new_labels = os.path.join(folder_new, "labels")
    new_images = os.path.join(folder_new, "images")
    output_labels = os.path.join(output_folder, "labels")
    output_images = os.path.join(output_folder, "images")

    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)

    # Get list of label filenames without extension
    annotated_files = {
        os.path.splitext(f)[0]
        for f in os.listdir(annotated_labels)
        if f.endswith(".txt")
    }
    new_files = {
        os.path.splitext(f)[0] for f in os.listdir(new_labels) if f.endswith(".txt")
    }

    # Find unmatched files
    unmatched_files = new_files - annotated_files

    # Copy unmatched labels and images
    for file in unmatched_files:
        label_path = os.path.join(new_labels, file + ".txt")
        image_path_jpg = os.path.join(new_images, file + ".jpg")
        image_path_png = os.path.join(new_images, file + ".png")

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_labels, file + ".txt"))

        if os.path.exists(image_path_jpg):
            shutil.copy(image_path_jpg, os.path.join(output_images, file + ".jpg"))
        elif os.path.exists(image_path_png):
            shutil.copy(image_path_png, os.path.join(output_images, file + ".png"))


if __name__ == "__main__":
    # copy_unmatched_files("SAR_manual_annotations", "temp_april", "SAR_April_CVAT")
    # convert_dataset_to_cvat_old(
    #     "SAR_April_CVAT/labels", "SAR_April_CVAT/images", "SAR(update)"
    # )
    # organize_files("biigle_extracted_annotations_april", "temp_april")
    # organize_images_and_labels(
    #     "update(SAR)_cvatExport/obj_train_data/obj_train_data",
    #     "SAR_April_CVAT/images",
    #     "reannotated_SAR_april_raw",
    # )
    # organize_files(
    #     "round_3_cvat_preparation/SAR_2025",
    #     "round_3_cvat_preparation/SAR_2025_organized",
    # )
    # convert_dataset_to_cvat_old(
    #     "round_3_cvat_preparation/AB_2022_organized/labels",
    #     "round_3_cvat_preparation/AB_2022_organized/images",
    #     "round_3_AB_2022",
    # )
    organize_images_and_labels(
        "round_3_cvat_export/AB_22_export/obj_train_data/obj_train_data",
        "round_3_cvat_preparation/AB_2022_organized/images",
        "round_3_cvat_export/AB_22_organized",
    )
