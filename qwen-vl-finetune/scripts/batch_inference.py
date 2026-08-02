import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_vl_utils import process_vision_info
from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
    Qwen2_5_VL_Nautilus_ForConditionalGeneration,
)

try:
    from tqdm import tqdm
except (
    ImportError
):  # pragma: no cover - tqdm should be available, but fall back gracefully

    def tqdm(iterable, **kwargs):
        return iterable


image_token_id = 151655
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scale_bboxes_in_text(text: str, w_scale: float, h_scale: float) -> str:
    def scale_bbox(match):
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        new_x1 = round(x1 * w_scale)
        new_y1 = round(y1 * h_scale)
        new_x2 = round(x2 * w_scale)
        new_y2 = round(y2 * h_scale)
        return f"[{new_x1}, {new_y1}, {new_x2}, {new_y2}]"

    pattern = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"
    scaled_text = re.sub(pattern, scale_bbox, text)
    return scaled_text


def get_grid_thw(processor, image_file):
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, List):
        image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]
    return grid_thw, width, height


def double_image_tokens(inputs: dict, image_token_id: int):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_ids = input_ids.squeeze(0)  # flatten to 1D
    attention_mask = attention_mask.squeeze(0)
    new_ids, new_mask = [], []

    for token, mask in zip(input_ids, attention_mask):
        new_ids.append(token.item())
        new_mask.append(mask.item())
        if token.item() == image_token_id:
            new_ids.append(token.item())
            new_mask.append(mask.item())

    return torch.tensor(
        new_ids, dtype=input_ids.dtype, device=input_ids.device
    ).unsqueeze(0), torch.tensor(
        new_mask, dtype=attention_mask.dtype, device=attention_mask.device
    ).unsqueeze(0)


def find_images(image_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in image_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_metadata(metadata_path: Path, prompt: str, checkpoint: str) -> dict:
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata.setdefault("image_dims", {})
    else:
        metadata = {"prompt": prompt, "checkpoint": checkpoint, "image_dims": {}}
    # Always reflect the prompt/checkpoint of the current run.
    metadata["prompt"] = prompt
    metadata["checkpoint"] = checkpoint
    return metadata


def save_metadata(metadata_path: Path, metadata: dict) -> None:
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


@torch.inference_mode()
def run_inference(
    model, processor, image_path: Path, prompt: str, max_new_tokens: int
) -> Tuple[str, int, int]:
    image_processor = processor.image_processor

    # Scale the bounding box for region captioning or other region-based questions
    # that include pixel coordinates in the prompt.
    grid_thw, ori_w, ori_h = get_grid_thw(image_processor, str(image_path))
    input_height = grid_thw[1].item() * 14
    input_width = grid_thw[2].item() * 14
    scale_h, scale_w = input_height / ori_h, input_width / ori_w
    scaled_prompt = scale_bboxes_in_text(prompt, scale_w, scale_h)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": scaled_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs["input_ids"], inputs["attention_mask"] = double_image_tokens(
        inputs, image_token_id
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    res_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return res_text, input_height, input_width


def main():
    parser = argparse.ArgumentParser(
        description="Run NAUTILUS (Qwen) inference over every image in a directory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save per-image .txt outputs.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image",
        help="Prompt to use for every image.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Max new tokens to generate."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of --image-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output .txt already exists in --output-dir.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device (0-3)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    device = "cuda:" + args.device

    images = find_images(image_dir, args.recursive)
    if not images:
        print(f"No images found in {image_dir}")
        return
    print(f"Found {len(images)} images in {image_dir}")

    metadata = load_metadata(metadata_path, args.prompt, args.checkpoint)

    # Load Model (once)
    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Set Min/Max Pixel Size
    min_pixels = 1 * 28 * 28
    max_pixels = 1338 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
    )

    num_ok, num_skipped, num_failed = 0, 0, 0
    for image_path in tqdm(images, desc="Running inference"):
        out_path = results_dir / (image_path.stem + ".txt")
        if args.skip_existing and out_path.exists():
            num_skipped += 1
            continue

        try:
            res_text, input_height, input_width = run_inference(
                model, processor, image_path, args.prompt, args.max_new_tokens
            )
        except Exception as e:  # noqa: BLE001
            print(f"[FAILED] {image_path}: {e}")
            num_failed += 1
            continue

        out_path.write_text(res_text)
        metadata["image_dims"][image_path.name] = {
            "input_height": input_height,
            "input_width": input_width,
        }
        save_metadata(metadata_path, metadata)
        num_ok += 1

    print(
        f"Done. Success: {num_ok}, Skipped: {num_skipped}, Failed: {num_failed}. "
        f"Outputs written to {results_dir}, metadata written to {metadata_path}"
    )


if __name__ == "__main__":
    main()
