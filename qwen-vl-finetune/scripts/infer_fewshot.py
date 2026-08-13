"""Few-shot in-context inference for Nautilus-Qwen.

Feeds a set of exemplar images (each with a textual description) followed by a
query image, so the model is asked to generalise from the exemplars rather than
from its closed training label set.

Exemplars are supplied as a JSON file:

    [
      {"image": "/path/ex1.jpg", "description": "The target species is the small orange fish."},
      {"image": "/path/ex2.jpg", "description": "Another view of the target species."}
    ]

Note: Nautilus doubles the vision tokens of every image, so exemplars are
rendered at a lower resolution than the query image to stay inside
model_max_length (8192). See --exemplar_max_pixels / --query_max_pixels.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_vl_utils import process_vision_info
from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
    Qwen2_5_VL_Nautilus_ForConditionalGeneration,
)

image_token_id = 151655
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_PROMPT = (
    "If any species of the type shown in the example images is present in this "
    "image, detect them and return the location of each as a bounding box in "
    'JSON format, e.g. {"bbox_2d": [x1, y1, x2, y2], "label": "..."}. '
    "If none are present, answer exactly: none."
)

BBOX_PATTERN = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"


def scale_bboxes_in_text(text: str, w_scale: float, h_scale: float) -> str:
    """Rescale every [x1, y1, x2, y2] occurrence in `text`."""

    def scale_bbox(match):
        x1, y1, x2, y2 = (int(match.group(i)) for i in range(1, 5))
        return (
            f"[{round(x1 * w_scale)}, {round(y1 * h_scale)}, "
            f"{round(x2 * w_scale)}, {round(y2 * h_scale)}]"
        )

    return re.sub(BBOX_PATTERN, scale_bbox, text)


def double_image_tokens(inputs: dict, image_token_id: int):
    """Duplicate every image placeholder: the VFE branch emits 2x image features."""
    input_ids = inputs["input_ids"].squeeze(0)
    attention_mask = inputs["attention_mask"].squeeze(0)
    new_ids, new_mask = [], []

    for token, mask in zip(input_ids, attention_mask):
        new_ids.append(token.item())
        new_mask.append(mask.item())
        if token.item() == image_token_id:
            new_ids.append(token.item())
            new_mask.append(mask.item())

    return (
        torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0),
        torch.tensor(new_mask, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0),
    )


def build_messages(exemplars, query_path, prompt, exemplar_max_pixels, query_max_pixels, intro):
    """Interleave exemplar image/description pairs, then the query image + prompt."""
    content = [{"type": "text", "text": intro}]

    for i, ex in enumerate(exemplars, start=1):
        content.append(
            {"type": "image", "image": ex["image"], "max_pixels": exemplar_max_pixels}
        )
        content.append({"type": "text", "text": f"Example {i}: {ex['description']}"})

    content.append({"type": "text", "text": "Now consider this new image:"})
    content.append(
        {"type": "image", "image": str(query_path), "max_pixels": query_max_pixels}
    )
    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--exemplars", required=True, help="JSON file with [{image, description}, ...].")
    parser.add_argument("--query_dir", help="Directory of query images.")
    parser.add_argument("--query", help="Single query image (alternative to --query_dir).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction applied to the query image.")
    parser.add_argument(
        "--intro",
        default="Here are example images of a target species:",
        help="Text shown before the exemplars.",
    )
    parser.add_argument(
        "--exemplar_max_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Pixel budget per exemplar image (kept low: tokens are doubled).",
    )
    parser.add_argument(
        "--query_max_pixels",
        type=int,
        default=1338 * 28 * 28,
        help="Pixel budget for the query image (training-time default).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--model_max_length", type=int, default=8192, help="Used only to warn about overflow.")
    parser.add_argument(
        "--raw_coords",
        action="store_true",
        help="Report bboxes in model input space instead of original query-image pixels.",
    )
    parser.add_argument("--output", help="Optional path to save results as JSON.")
    args = parser.parse_args()

    if not args.query_dir and not args.query:
        parser.error("provide --query_dir or --query")

    with open(args.exemplars) as f:
        exemplars = json.load(f)
    for ex in exemplars:
        if "image" not in ex or "description" not in ex:
            parser.error(f"exemplar entries need 'image' and 'description' keys, got: {ex}")
    print(f"Loaded {len(exemplars)} exemplars from {args.exemplars}")

    if args.query_dir:
        query_paths = sorted(
            p for p in Path(args.query_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not query_paths:
            print(f"No images found in {args.query_dir}")
            return
    else:
        query_paths = [Path(args.query)]
    print(f"Found {len(query_paths)} query images")

    print("Loading model...")
    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    # Give the processor the larger budget so per-image limits set above are not
    # overridden; qwen_vl_utils has already resized each image by this point.
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=1 * 28 * 28, max_pixels=args.query_max_pixels
    )

    results = []
    warned = False
    for query_path in tqdm(query_paths):
        messages = build_messages(
            exemplars, query_path, args.prompt,
            args.exemplar_max_pixels, args.query_max_pixels, args.intro,
        )

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

        # The query image is the last one passed in.
        query_grid = inputs["image_grid_thw"][-1]
        input_height = query_grid[1].item() * 14
        input_width = query_grid[2].item() * 14

        # Prompt bboxes (if any) refer to the query image -> scale into model space.
        if re.search(BBOX_PATTERN, args.prompt):
            with Image.open(query_path) as img:
                ori_w, ori_h = img.size
            scaled_prompt = scale_bboxes_in_text(
                args.prompt, input_width / ori_w, input_height / ori_h
            )
            messages = build_messages(
                exemplars, query_path, scaled_prompt,
                args.exemplar_max_pixels, args.query_max_pixels, args.intro,
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )

        inputs["input_ids"], inputs["attention_mask"] = double_image_tokens(inputs, image_token_id)

        n_tokens = inputs["input_ids"].shape[1]
        if not warned and n_tokens + args.max_new_tokens > args.model_max_length:
            print(
                f"\nWARNING: prompt is {n_tokens} tokens and max_new_tokens="
                f"{args.max_new_tokens} exceeds model_max_length={args.model_max_length}. "
                f"Lower --exemplar_max_pixels or use fewer exemplars."
            )
            warned = True

        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Model emits coordinates in input space; map back to original pixels.
        with Image.open(query_path) as img:
            ori_w, ori_h = img.size
        if args.raw_coords:
            rescaled = response
        else:
            rescaled = scale_bboxes_in_text(response, ori_w / input_width, ori_h / input_height)

        results.append({
            "query_image": str(query_path),
            "prompt": args.prompt,
            "n_exemplars": len(exemplars),
            "prompt_tokens": n_tokens,
            "input_size": [input_width, input_height],
            "original_size": [ori_w, ori_h],
            "response": rescaled,
            "response_raw_coords": response,
        })
        print(f"\n{query_path.name} [{n_tokens} tok]: {rescaled}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
