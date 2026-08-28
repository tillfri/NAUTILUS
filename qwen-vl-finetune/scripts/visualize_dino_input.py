"""Render the image DINOv2 actually sees inside NAUTILUS' VFE.

Two defects are baked into the trained weights (see
``masterarbeit/nautilus-model-architecture.md`` "Two things that look like bugs"):

1. Qwen's image processor emits patches in **2x2-merge-block order**
   (``Qwen2VLImageProcessor._preprocess``'s ``transpose(0,3,6,4,7,2,1,5,8)``), but
   ``Qwen2_5_VL_Nautilus...restore_image_from_patches`` reassembles them assuming
   plain raster order. Every 2x2 patch block is therefore smeared into a 1x4
   horizontal strip: the image stays coarsely correct but is locally shredded.
2. DINOv2 is fed Qwen's CLIP-style normalisation (mean 0.481/0.458/0.408,
   std 0.269/0.261/0.276), not its own ImageNet statistics.

``dino_view()`` reproduces both on a normal RGB image, torch-free (numpy + PIL),
so you can see exactly what the frozen ViT-L is handed.

Usage (host, no GPU needed):

    python3 visualize_dino_input.py /path/to/image.jpg --out /tmp/dino_view.png

Usage (container, identical result):

    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts && \
       python3 visualize_dino_input.py /workspace/datasets/thuenen_scaling/images/test/<frame>.png \
         --out /workspace/runs/dino_view.png"
"""

import argparse
import math
import os

import numpy as np
from PIL import Image

# Qwen2-VL preprocessing constants (transformers 4.51.3, matches the thesis launcher).
PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28
DEFAULT_MIN_PIXELS = 1 * FACTOR * FACTOR          # 784, launcher value
DEFAULT_MAX_PIXELS = 1338 * FACTOR * FACTOR       # 1_048_992, launcher value

# Qwen's normalisation (the one DINOv2 wrongly receives).
QWEN_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)
QWEN_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float64)


def smart_resize(height, width, factor=FACTOR,
                 min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    """Port of ``Qwen2VLImageProcessor.smart_resize``: nearest multiple of
    ``factor`` on each side, aspect ratio preserved, area clamped to
    ``[min_pixels, max_pixels]``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio must be < 200:1")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def scramble_patches(img_hw3):
    """Apply defect #1 only: the merge-block -> raster patch permutation.

    ``img_hw3`` is a float array (H, W, 3) whose H and W are multiples of
    ``PATCH_SIZE``. Returns an array of the same shape with every 2x2 block of
    14px patches unrolled into a 1x4 horizontal strip -- exactly what
    ``restore_image_from_patches`` produces from Qwen's patch sequence.
    """
    h, w, c = img_hw3.shape
    gh, gw = h // PATCH_SIZE, w // PATCH_SIZE
    assert gh % MERGE_SIZE == 0 and gw % MERGE_SIZE == 0, "grid must be even (factor=28)"
    m, p = MERGE_SIZE, PATCH_SIZE

    # (gh, gw, p, p, c) grid of patches, raster order.
    patches = (img_hw3
               .reshape(gh, p, gw, p, c)
               .transpose(0, 2, 1, 3, 4))

    # Qwen sequence order: (block_row, block_col, in_row, in_col).
    seq = (patches
           .reshape(gh // m, m, gw // m, m, p, p, c)
           .transpose(0, 2, 1, 3, 4, 5, 6)
           .reshape(gh * gw, p, p, c))

    # restore_image_from_patches: read that sequence back as raster (gh, gw).
    out = (seq
           .reshape(gh, gw, p, p, c)
           .transpose(0, 2, 1, 3, 4)
           .reshape(gh * p, gw * p, c))
    return out


def dino_view(image_path, apply_scramble=True, apply_norm=True,
              min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    """Return (resized_reference, dino_input) as uint8 RGB arrays.

    ``resized_reference`` is the frame after ``smart_resize`` -- what the model
    would see with a correct pipeline. ``dino_input`` is what the frozen DINOv2
    actually receives: patch-scrambled (defect #1) and, for display, the
    CLIP-normalised tensor mapped back to 0..255 by inverting *DINOv2's* expected
    ImageNet stats, so the residual colour cast from defect #2 is visible.
    """
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    h, w = smart_resize(h0, w0, min_pixels=min_pixels, max_pixels=max_pixels)
    resized = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float64) / 255.0

    view = resized
    if apply_scramble:
        view = scramble_patches(view)

    if apply_norm:
        # DINOv2 gets Qwen-normalised pixels; visualise by de-normalising with the
        # ImageNet stats DINOv2 *thinks* it received.
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        normed = (view - QWEN_MEAN) / QWEN_STD
        view = normed * imagenet_std + imagenet_mean

    ref_u8 = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    view_u8 = np.clip(view * 255.0, 0, 255).astype(np.uint8)
    return ref_u8, view_u8


def _side_by_side(ref_u8, view_u8, gap=12):
    h = max(ref_u8.shape[0], view_u8.shape[0])
    def pad(a):
        out = np.zeros((h, a.shape[1], 3), dtype=np.uint8)
        out[:a.shape[0]] = a
        return out
    strip = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.concatenate([pad(ref_u8), strip, pad(view_u8)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="PNG path (default: <image>.dino_view.png)")
    ap.add_argument("--no-scramble", action="store_true", help="skip defect #1")
    ap.add_argument("--no-norm", action="store_true", help="skip defect #2")
    ap.add_argument("--only-view", action="store_true",
                    help="write just the DINO input, not the side-by-side")
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = ap.parse_args()

    ref_u8, view_u8 = dino_view(
        args.image,
        apply_scramble=not args.no_scramble,
        apply_norm=not args.no_norm,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    out = args.out or (os.path.splitext(args.image)[0] + ".dino_view.png")
    result = view_u8 if args.only_view else _side_by_side(ref_u8, view_u8)
    Image.fromarray(result).save(out)
    print(f"grid {ref_u8.shape[0] // PATCH_SIZE}x{ref_u8.shape[1] // PATCH_SIZE} patches (h x w)"
          f" -> {out}  ({'left: smart_resize reference | right: DINO input' if not args.only_view else 'DINO input'})")


if __name__ == "__main__":
    main()
