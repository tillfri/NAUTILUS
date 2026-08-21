"""Box proposals from Megalodon and NAUTILUS over *every* Thünen split.

Feeds the annotation-refinement pipeline: the Thünen ground truth is known to be
both too loose (3386 of 9293 BIIGLE shapes were drawn with the circle tool and
``annotations.py:shape_to_box`` turns a circle into its bounding **square**, so GT
boxes are ~2.09x the area of a tight box) and too sparse (a large share of clearly
visible animals were never annotated). ``merge_annotations.py`` fixes both by
replacing GT geometry with a contained proposal and adding the missed instances --
this script produces the proposals it consumes.

Unlike ``megalodon_zeroshot.py`` / ``nautilus_zeroshot.py``, which run the **test**
split to produce a comparison number, this runs **train, val and test**, because the
output is an annotation source rather than a measurement.

Two models, two containers, one script:

* ``--model megalodon`` -- frozen ``mbari-megalodon-yolov8x.pt`` at imgsz 1280 and
  **conf 0.01**. The low threshold is deliberate: the operating point is chosen later
  by ``merge_annotations.py --megalodon-conf``, which is a free in-memory filter over
  these boxes. Runs in the ``brackish`` container, which is the only place Ultralytics
  lives. Boxes come out in **original-image pixels**.
* ``--model nautilus`` -- the NAUTILUS checkpoint with the ``nautilus_zeroshot.py``
  prompt. Runs in ``nautilus-qwen``. Boxes come out in **model-input pixels** and only
  ``metadata.json``'s ``input_width``/``input_height`` can map them back.

Both write the canonical ``batch_inference.py`` layout, one run directory per split::

    <out>/<split>/results/<stem>.txt      one JSON object per line
    <out>/<split>/metadata.json           {"prompt", "checkpoint", "image_dims"}

Two things this script does differently from its predecessors, both on purpose:

* **The grid is read off the real inputs.** ``batch_inference.get_grid_thw`` runs the
  processor over the *original* PIL image while the model is actually fed
  ``process_vision_info``'s already-rounded output, so it records grid 54x96
  (``input_width`` 1344) for a 2704x1520 frame whose ``pixel_values`` carry 54x98
  (1372) patches -- a systematic ~2.1% under-report that inflates every rescaled
  x-coordinate downstream. Here ``input_height``/``input_width`` come from
  ``inputs["image_grid_thw"]``, the tensor the model actually saw, as
  ``prompt_experiments.py:run_messages`` does.
* **The query image still carries no ``max_pixels``.** That is the house path (see
  ``prompt_variants.query_turn``): passing it resizes in one step instead of two,
  which lands on the same grid but different pixels, and greedy decoding then
  diverges completely (measured 0/12 identical against a 12/12 same-GPU noise floor).
  Fixing the grid readout is a metadata fix; changing the resize would silently make
  these proposals a different model's.

NAUTILUS is ~1.6 s/image, so all 7203 frames take ~3.2 h on one GPU. ``--shard i/N``
splits the work across GPUs; each shard writes its own ``metadata.shard<i>.json``
(concurrent writers would corrupt a shared one) and ``--merge-metadata`` folds them
into the final ``metadata.json`` once the shards are done.

Usage:
    # 1. Megalodon, all splits, in the brackish container (needs build_proposals.py
    #    and yolo_scaling.py copied into ~/brackish/container/):
    docker exec brackish python3 /usr/src/ultralytics/brackish/build_proposals.py \
        --model megalodon \
        --root /usr/src/ultralytics/brackish/thuenen_scaling \
        --out  /usr/src/ultralytics/brackish/megalodon_proposals \
        --splits train,val,test --conf 0.01 --imgsz 1280 --device 0
    # the brackish container cannot see the NAUTILUS tree, so copy the text back:
    cp -r /home/tfricke/brackish/container/megalodon_proposals \
          /home/tfricke/nautilus/runs/megalodon_proposals

    # 2. NAUTILUS, all splits, sharded over beta's four GPUs:
    for i in 0 1 2 3; do
      docker exec -d nautilus-qwen bash -lc \
        "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \
         python3 build_proposals.py --model nautilus --splits train,val,test \
           --out /workspace/runs/nautilus_proposals --device $i --shard $i/4 \
           > /workspace/runs/nautilus_proposals_g$i.log 2>&1"
    done
    # once all four exit:
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/build_proposals.py \
        --model nautilus --out /workspace/runs/nautilus_proposals --merge-metadata
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_SPLITS = "train,val,test"

MEGALODON_ROOT = "/usr/src/ultralytics/brackish/thuenen_scaling"
MEGALODON_WEIGHTS = "/usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt"
MEGALODON_OUT = "/usr/src/ultralytics/brackish/megalodon_proposals"
# Low enough that the whole useful threshold range survives as a filter later.
MEGALODON_CONF = 0.01

NAUTILUS_ROOT = "/workspace/datasets/thuenen_scaling"
NAUTILUS_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"
NAUTILUS_OUT = "/workspace/runs/nautilus_proposals"


def parse_splits(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_shard(text):
    """``"2/4"`` -> ``(2, 4)``. ``None`` -> ``(0, 1)``, i.e. the whole split."""
    if not text:
        return 0, 1
    index, _, count = text.partition("/")
    index, count = int(index), int(count)
    if count < 1 or not 0 <= index < count:
        raise SystemExit("--shard must be i/N with 0 <= i < N, got {!r}".format(text))
    return index, count


# --------------------------------------------------------------------------- #
# Megalodon
# --------------------------------------------------------------------------- #

def run_megalodon(args):
    """One GPU pass per split at ``--conf``, written straight out unfiltered."""
    from ultralytics import YOLO

    from yolo_scaling import predict_split, write_nautilus_format

    model = YOLO(args.weights, task="detect")
    if len(model.names) != 1:
        print("[warn] {} classes in the checkpoint, expected the 1-class Megalodon"
              .format(len(model.names)))
    label_by_class_id = {class_id: args.label for class_id in model.names}

    for split in parse_splits(args.splits):
        out_dir = os.path.join(args.out, split)
        if os.path.exists(os.path.join(out_dir, "metadata.json")) and not args.force:
            print("[skip] {} already has metadata.json (--force to redo)".format(out_dir))
            continue
        os.makedirs(out_dir, exist_ok=True)

        print("\n=== megalodon: {} (conf {}, imgsz {}) ===".format(
            split, args.conf, args.imgsz))
        collected = predict_split(model, args.root, split, args.conf, args.device,
                                  chunk_size=args.pred_batch, imgsz=args.imgsz)
        results_dir, written = write_nautilus_format(
            out_dir, collected, label_by_class_id, args.weights,
            prompt="megalodon-proposals", min_score=0.0)
        per_image = [len(boxes) for _, _, _, boxes in collected]
        print("{} images, {} boxes ({:.2f}/image, max {}) -> {}".format(
            len(collected), written,
            written / len(collected) if collected else 0.0,
            max(per_image) if per_image else 0, results_dir))
    return 0


# --------------------------------------------------------------------------- #
# NAUTILUS
# --------------------------------------------------------------------------- #

def generate(model, processor, image_path, prompt, max_new_tokens):
    """One query image -> ``(response, input_height, input_width)``.

    The dimensions are read off ``inputs["image_grid_thw"]`` -- the tensor the model
    is actually fed -- not off a second, independent preprocess of the original
    image the way ``batch_inference.get_grid_thw`` does it. See the module docstring.
    """
    import torch

    from batch_inference import image_token_id
    from infer_fewshot import double_image_tokens
    from prompt_experiments import build_inputs
    from prompt_variants import single

    # ``single`` never touches its ctx argument; it exists to build exactly the
    # message shape the baseline run used -- image first, no max_pixels, then text.
    messages = single(None, str(image_path), prompt)
    _text, inputs = build_inputs(processor, messages)

    grid = inputs["image_grid_thw"][-1]
    input_height = int(grid[1].item()) * 14
    input_width = int(grid[2].item()) * 14

    inputs["input_ids"], inputs["attention_mask"] = double_image_tokens(
        inputs, image_token_id)
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out_ids[len(in_ids):]
               for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    return response, input_height, input_width


def metadata_name(shard_index, shard_count):
    """Sharded runs must not share one file -- concurrent writers would corrupt it."""
    if shard_count == 1:
        return "metadata.json"
    return "metadata.shard{}.json".format(shard_index)


def merge_metadata(args):
    """Fold every ``metadata.shard*.json`` of a split into one ``metadata.json``."""
    for split in parse_splits(args.splits):
        out_dir = os.path.join(args.out, split)
        shards = sorted(name for name in os.listdir(out_dir)
                        if name.startswith("metadata.shard") and name.endswith(".json"))
        if not shards:
            print("[skip] {}: no shard metadata to merge".format(out_dir))
            continue

        merged = None
        for name in shards:
            with open(os.path.join(out_dir, name), encoding="utf-8") as handle:
                part = json.load(handle)
            if merged is None:
                merged = {"prompt": part["prompt"], "checkpoint": part["checkpoint"],
                          "image_dims": {}}
            merged["image_dims"].update(part["image_dims"])

        path = os.path.join(out_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2)

        n_results = len([n for n in os.listdir(os.path.join(out_dir, "results"))
                         if n.endswith(".txt")])
        n_dims = len(merged["image_dims"])
        flag = "" if n_dims == n_results else "   <-- MISMATCH, a shard is incomplete"
        print("{}: {} shards -> {} image_dims against {} result files{}".format(
            split, len(shards), n_dims, n_results, flag))
    return 0


def run_nautilus(args):
    """Mirrors ``nautilus_zeroshot.run_zero_shot`` over every split, shardable."""
    sys.path.insert(0, os.path.dirname(HERE))
    import torch  # noqa: F401  (imported for its side effects on CUDA init)
    from PIL import Image
    from transformers import AutoProcessor
    from tqdm import tqdm

    from batch_inference import find_images
    from nautilus_zeroshot import PROMPT_TEMPLATE, snap_labels
    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_VL_Nautilus_ForConditionalGeneration,
    )
    from pathlib import Path

    with open(os.path.join(args.root, "classes_prompt.txt"), encoding="utf-8") as handle:
        prompt_names = [line.strip() for line in handle if line.strip()]
    prompt = args.prompt or PROMPT_TEMPLATE.format(classes=", ".join(prompt_names))
    print("prompt: {}".format(prompt))

    shard_index, shard_count = parse_shard(args.shard)
    if shard_count > 1:
        print("shard {} of {}".format(shard_index, shard_count))

    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:" + args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=1 * 28 * 28, max_pixels=args.max_pixels * 28 * 28)

    for split in parse_splits(args.splits):
        image_dir = os.path.join(args.root, split, "images")
        images = find_images(Path(image_dir), recursive=False)
        images = images[shard_index::shard_count]
        if args.limit:
            images = images[:args.limit]
        if not images:
            raise SystemExit("no images under {}".format(image_dir))

        out_dir = os.path.join(args.out, split)
        results_dir = os.path.join(out_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        metadata_path = os.path.join(out_dir, metadata_name(shard_index, shard_count))

        metadata = {"prompt": prompt, "checkpoint": args.checkpoint, "image_dims": {}}
        if args.skip_existing and os.path.exists(metadata_path):
            # Resume: keep the dimensions already recorded, they cannot be recovered
            # from the results text alone.
            with open(metadata_path, encoding="utf-8") as handle:
                metadata["image_dims"] = json.load(handle)["image_dims"]

        ok = skipped = failed = unmatched_labels = 0
        description = "{} [{}/{}]".format(split, shard_index, shard_count)
        for image_path in tqdm(images, desc=description):
            out_path = os.path.join(results_dir, image_path.stem + ".txt")
            if (args.skip_existing and os.path.exists(out_path)
                    and image_path.name in metadata["image_dims"]):
                skipped += 1
                continue
            try:
                response, input_height, input_width = generate(
                    model, processor, image_path, prompt, args.max_new_tokens)
            except Exception as error:  # noqa: BLE001
                print("[failed] {}: {}".format(image_path, error))
                failed += 1
                continue

            snapped, unmatched = snap_labels(response, prompt_names)
            unmatched_labels += unmatched
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(snapped)
            with Image.open(image_path) as image:
                width, height = image.size
            metadata["image_dims"][image_path.name] = {
                "input_height": input_height,
                "input_width": input_width,
                "original_width": width,
                "original_height": height,
            }
            ok += 1
            if ok % 50 == 0:
                with open(metadata_path, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print("{}: ok={} skipped={} failed={}, {} labels did not map onto a prompt "
              "name -> {}".format(split, ok, skipped, failed, unmatched_labels,
                                  metadata_path))

    if shard_count > 1:
        print("\nshards done; fold the metadata with:\n"
              "  python3 {} --model nautilus --out {} --splits {} --merge-metadata"
              .format(os.path.abspath(__file__), args.out, args.splits))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["megalodon", "nautilus"])
    parser.add_argument("--splits", default=DEFAULT_SPLITS,
                        help="Comma-separated splits to run (default: all three).")
    parser.add_argument("--root", default=None,
                        help="Dataset root; defaults per model to the container path.")
    parser.add_argument("--out", default=None,
                        help="Run root; one sub-directory per split.")
    parser.add_argument("--device", default="0", help="CUDA device index.")
    parser.add_argument("--force", action="store_true",
                        help="megalodon: redo splits that already have metadata.json.")

    megalodon = parser.add_argument_group("megalodon")
    megalodon.add_argument("--weights", default=MEGALODON_WEIGHTS)
    megalodon.add_argument("--conf", type=float, default=MEGALODON_CONF,
                           help="Detector threshold; keep it low, it is filtered later.")
    megalodon.add_argument("--imgsz", type=int, default=1280,
                           help="Megalodon's training resolution.")
    megalodon.add_argument("--label", default="object",
                           help="Constant label; evaluate_detections.py drops boxes "
                                "without one.")
    megalodon.add_argument("--pred-batch", type=int, default=16)

    nautilus = parser.add_argument_group("nautilus")
    nautilus.add_argument("--checkpoint", default=NAUTILUS_CHECKPOINT)
    nautilus.add_argument("--prompt", default=None,
                          help="Overrides the all-classes detection prompt.")
    nautilus.add_argument("--max-new-tokens", type=int, default=1024)
    nautilus.add_argument("--max-pixels", type=int, default=1338,
                          help="Processor budget in 28x28 patches (training value).")
    nautilus.add_argument("--shard", default=None, metavar="i/N",
                          help="Run only every N-th image, offset i.")
    nautilus.add_argument("--skip-existing", action="store_true")
    nautilus.add_argument("--limit", type=int, default=None,
                          help="First N images of each shard, for a smoke test.")
    nautilus.add_argument("--merge-metadata", action="store_true",
                          help="Fold metadata.shard*.json into metadata.json and exit.")

    args = parser.parse_args()
    if args.model == "megalodon":
        args.root = args.root or MEGALODON_ROOT
        args.out = args.out or MEGALODON_OUT
        return run_megalodon(args)

    args.root = args.root or NAUTILUS_ROOT
    args.out = args.out or NAUTILUS_OUT
    if args.merge_metadata:
        return merge_metadata(args)
    return run_nautilus(args)


if __name__ == "__main__":
    raise SystemExit(main())
