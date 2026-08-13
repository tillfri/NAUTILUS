import base64
import json
import os
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


def compare_yolo_annotations(
    dir1, dir2, output_json="annoations_diffs_AnnotationRound1_SAR"
):
    removed_image = []
    removed_annotations = []
    additional_annotations = []

    files1 = [f for f in os.listdir(dir1) if f.endswith(".txt")]
    files2_set = set(f for f in os.listdir(dir2) if f.endswith(".txt"))

    result = {}

    for file1 in files1:
        file1_path = os.path.join(dir1, file1)
        file2_path = os.path.join(dir2, file1)

        if file1 not in files2_set:
            removed_image.append(file1)
            result[file1] = {"status": "removed_image"}
        else:
            with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

                if len(lines1) == len(lines2):
                    continue  # Done
                elif len(lines1) < len(lines2):
                    additional_annotations.append(file1)
                    result[file1] = {"status": "additional_annotations"}
                else:
                    removed_annotations.append(file1)
                    result[file1] = {"status": "removed_annotations"}

    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {output_json}")
    return removed_image, removed_annotations, additional_annotations


def draw_yolo_boxes_on_image(image_path, label_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, w, h = map(float, parts)
            x1 = (x_center - w / 2) * width
            y1 = (y_center - h / 2) * height
            x2 = (x_center + w / 2) * width
            y2 = (y_center + h / 2) * height
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # image.save("/home/till/Downloads/brackish/test.png")

    return image


def generate_comparison_grid(
    annotation_json,
    labels_dir1,
    labels_dir2,
    images_dir,
    output_path="full_comparison_grid.jpg",
):
    with open(annotation_json, "r") as f:
        diffs = json.load(f)

    comparison_rows = []

    try:
        font = ImageFont.truetype(
            "arial.ttf", 24
        )  # You can change to another font if needed
    except IOError:
        font = ImageFont.load_default()

    for filename, info in diffs.items():
        status = info.get("status")
        if status not in [
            "removed_annotations",
            "additional_annotations",
            "removed_image",
        ]:
            continue

        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(images_dir, base_name + ".jpg")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        label1_path = os.path.join(labels_dir1, filename)
        label2_path = os.path.join(labels_dir2, filename)

        if status == "removed_image":
            if os.path.exists(label1_path):
                left_img = draw_yolo_boxes_on_image(image_path, label1_path)
                right_img = Image.new("RGB", left_img.size, color=(60, 60, 60))
            elif os.path.exists(label2_path):
                right_img = draw_yolo_boxes_on_image(image_path, label2_path)
                left_img = Image.new("RGB", right_img.size, color=(60, 60, 60))
            else:
                print(f"No label file found for removed_image: {filename}")
                continue
        else:
            left_img = draw_yolo_boxes_on_image(image_path, label1_path)
            right_img = draw_yolo_boxes_on_image(image_path, label2_path)

        # Create the comparison row
        row_height = max(left_img.height, right_img.height)
        row_width = left_img.width + right_img.width
        row = Image.new("RGB", (row_width, row_height), color=(0, 0, 0))
        row.paste(left_img, (0, 0))
        row.paste(right_img, (left_img.width, 0))

        # Add status text above the row
        text_height = 30
        labeled_row = Image.new(
            "RGB", (row_width, row_height + text_height), color=(30, 30, 30)
        )
        labeled_row.paste(row, (0, text_height))

        draw = ImageDraw.Draw(labeled_row)
        draw.text((10, 5), f"{status} - {filename}", fill=(255, 255, 255), font=font)

        comparison_rows.append(labeled_row)

    if not comparison_rows:
        print("No differences found to visualize.")
        return

    # Compute final canvas size
    total_width = max(row.width for row in comparison_rows)
    total_height = sum(row.height for row in comparison_rows)

    full_image = Image.new("RGB", (total_width, total_height), color=(0, 0, 0))
    y_offset = 0
    for row in comparison_rows:
        full_image.paste(row, (0, y_offset))
        y_offset += row.height

    full_image.save(output_path)
    print(f"Saved full comparison grid to: {output_path}")


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_validation_doc(json_path, label_dir, image_dir, output_html_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for filename, url in data.items():
        label_file = os.path.join(label_dir, filename)
        image_file = os.path.join(
            image_dir, filename.replace(".txt", ".jpg")
        )  # assumes .jpg, adjust if needed

        if os.path.exists(label_file) and os.path.exists(image_file):
            try:
                image = draw_yolo_boxes_on_image(image_file, label_file)
                img_data = image_to_base64(image)
                row = f"""
                <tr>
                    <td><img src="{img_data}" width="400"></td>
                    <td><a href="{url}" target="_blank">{url}</a></td>
                    <td><input type="checkbox" name="{filename}" value="yes"> Valid</td>
                </tr>
                """
                rows.append(row)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Missing label or image file for: {filename}")

    html_content = f"""
    <html>
    <head>
        <title>YOLO Annotation Validation</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            td {{
                border: 1px solid #ccc;
                padding: 10px;
                vertical-align: top;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>YOLO Annotation Validation Document</h1>
        <form id="validationForm">
        <table>
            <tr><th>Image with Boxes</th><th>Annotation Link</th><th>Valid?</th></tr>
            {"".join(rows)}
        </table>
        </form>
        <br>
        <button onclick="downloadResults()">Download Results</button>

        <script>
        function downloadResults() {{
            const form = document.getElementById('validationForm');
            const data = {{}};
            for (const element of form.elements) {{
                if (element.type === 'checkbox') {{
                    data[element.name] = element.checked;
                }}
            }}
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type : 'application/json'}});
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'validation_results.json';
            link.click();
        }}
        </script>
    </body>
    </html>
    """
    with open(output_html_path, "w") as f:
        f.write(html_content)
    print(f"Validation document saved to {output_html_path}")


if __name__ == "__main__":
    # generate_validation_doc(
    #     json_path="additional_annotations_links.json",
    #     label_dir="update(SAR)_cvatExport/obj_train_data/obj_train_data",
    #     image_dir="SAR_April_CVAT/images",
    #     output_html_path="herman_checklist.html",
    # )
    # compare_yolo_annotations(
    #     "dataset_original_annotations/labels", "annotation_round1_SAR"
    # )
    # draw_yolo_boxes_on_image(
    #     "/home/till/Downloads/brackish/images/2019-03-25_23-17-56to2019-03-25_23-18-04_1-0106.png",
    #     "/home/till/Downloads/brackish/2019-03-25_23-17-56to2019-03-25_23-18-04_1-0106.txt",
    # )
    # compare_yolo_annotations(
    #     "round_3_cvat_preparation/AB_2022_organized/labels",
    #     "round_3_cvat_export/AB_22_export/obj_train_data/obj_train_data",
    #     output_json="annotation_diffs_round_3_ab_22",
    # )
    # generate_comparison_grid(
    #     "annoations_diffs_AnnotationRound1_SAR",
    #     "dataset_original_annotations/labels",
    #     "annotation_round1_SAR",
    #     "dataset_original_annotations/images",
    #     output_path="comparison_annotationRound1_SAR.jpg",
    # )
    generate_comparison_grid(
        "annotation_diffs_round_3_sar_25",
        "round_3_cvat_preparation/SAR_2025_organized/labels",
        "round_3_cvat_export/SAR_25_export/obj_train_data/obj_train_data",
        "round_3_cvat_preparation/SAR_2025_organized/images",
        output_path="round_3_sar_25_comparison_grid.jpg",
    )
