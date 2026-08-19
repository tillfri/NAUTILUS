"""Screening sweep: does NAUTILUS obey the class list it is given?

The Thünen zero-shot run collapsed every label onto ~4 words of its own training
vocabulary -- 960 of 1571 predictions said "starfish", 17 of the 24 prompt classes
were never emitted, and "brittle star" (the most frequent GT class, 407 boxes) got
zero predictions. Class-aware mAP@0.5 was 0.0007 against a class-agnostic 0.0108, a
15x drop purely from naming.

This script screens 12 prompt and few-shot variants (``prompt_variants.py``) over a
fixed ~250-image subsample of the test split. The primary signal is **prompt
adherence**, not mAP: at 0.0007 mAP has no resolving power, but "how many predicted
labels are even in the prompt vocabulary" does.

Three modes
-----------
``--make-subsample``
    Draw the screening subsample and pin it in ``screening_subsample.json``. Needs
    no GPU and no torch. Two stages: a rarest-first class fill (so the tail classes
    have non-zero support) then a video-stratified draw to the target size. Written
    once and then left alone -- same discipline as ``prompt_names.csv``.

``--make-exemplars``
    Pick the few-shot exemplar frames from the **train** split and write
    ``exemplars_k<K>.json``. Also torch-free. Splits are by video, so a train
    exemplar never shares a video with a test query and there is no
    near-duplicate-frame leakage. The sets are nested (k=2 subset of k=4 subset of
    k=8) so the F1/F2/F3 trend is a scaling curve rather than three unrelated draws.

default (inference)
    Run one or more named variants over the subsample and write
    ``batch_inference``-layout output per variant.

Output layout, ``<runs>/<variant>/``
------------------------------------
``results/<stem>.txt``
    ``snap_labels``-rewritten text, for ``evaluate_detections.py``.
``results_raw/<stem>.txt``
    The **unsnapped** response. ``snap_labels`` canonicalises case and does
    substring matching, so it would let e.g. "sea star" collapse onto "sand star";
    the in-vocabulary share -- the primary metric -- cannot be recovered from the
    snapped text. Both files are written for every image.
``metadata.json``
    The ``batch_inference`` keys (``prompt``, ``checkpoint``, ``image_dims``) plus
    ``variant``, ``subsample_sha1``, ``exemplars``, the pixel budgets and per-image
    ``prompt_tokens``.

Coordinates in ``results/`` are in **model-input pixel space**, exactly as
``batch_inference.py`` writes them; ``evaluate_detections.py`` rescales them back
using ``image_dims``. Do not rescale here.

Usage:
    # 1. draw the subsample (no GPU, no torch)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_experiments.py --make-subsample

    # 2. pick the few-shot exemplars (no GPU, no torch)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_experiments.py --make-exemplars

    # 3. run a shard of variants on one GPU (model is loaded once for the shard)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_experiments.py \
      --variants P0_baseline,P1_closed_vocab,P2_agnostic --device 0

    # 4. smoke-test one few-shot variant and dump the rendered chat template
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_experiments.py \
      --variants F1_icl_k2 --limit 5 --dump-text --device 0

    # 5. score one variant (subsample scores against the full test GT unchanged --
    #    evaluate_detections.py intersects the stems and skips the rest)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_prompt/P0_baseline/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_prompt/P0_baseline/metrics.json
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

# Both are torch-free at module scope (nautilus_zeroshot defers torch into
# run_zero_shot, yolo_scaling defers ultralytics into its training helpers), so
# --make-subsample and --make-exemplars stay importable on a host without torch.
from nautilus_zeroshot import video_id_of  # noqa: E402
from yolo_scaling import select_images  # noqa: E402

DEFAULT_ROOT = "/workspace/datasets/thuenen_scaling"
DEFAULT_RUNS = "/workspace/runs/thuenen_prompt"
DEFAULT_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"
SUBSAMPLE_NAME = "screening_subsample.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ── dataset reading ──────────────────────────────────────────────────────────
def read_prompt_classes(root):
    """The prompt-space class names, line index = class id."""
    with open(os.path.join(root, "classes_prompt.txt"), encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_boxes(root, split, label_dir="labels_prompt"):
    """Read one split's YOLO labels.

    ``yolo_scaling.load_split`` does this too, but only for the *train* split in the
    *fine* label space. The screening sweep needs an arbitrary split in the prompt
    space, so the loop is repeated here rather than bent out of shape.

    Returns:
        ``{image_name: [(class_id, cx, cy, w, h), ...]}`` over every image that has
        a file in ``<split>/images``, including images with no boxes.
    """
    images_dir = os.path.join(root, split, "images")
    labels_dir = os.path.join(root, split, label_dir)
    boxes = {}
    for image_name in sorted(os.listdir(images_dir)):
        if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        found = []
        if os.path.isfile(label_path):
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 5:
                        found.append((int(float(parts[0])),) + tuple(float(v) for v in parts[1:5]))
        boxes[image_name] = found
    return boxes


def classes_of(boxes):
    """``{image_name: {class_id, ...}}`` -- the shape ``select_images`` expects."""
    return {name: {box[0] for box in bs} for name, bs in boxes.items()}


def support_histograms(boxes, names, subset=None):
    """Per-class ``{boxes, images}`` counts over all of ``boxes`` or a subset."""
    keys = sorted(boxes) if subset is None else sorted(subset)
    box_counts, image_counts = Counter(), Counter()
    for name in keys:
        seen = set()
        for box in boxes[name]:
            box_counts[box[0]] += 1
            seen.add(box[0])
        for class_id in seen:
            image_counts[class_id] += 1
    return {names[c]: {"boxes": box_counts[c], "images": image_counts[c]}
            for c in range(len(names))}


def fingerprint(stems):
    """sha1 over the newline-joined sorted stems.

    Same idiom as ``yolo_init_comparison.slice_fingerprint``: the subsample is the
    thing every variant must share, and it is shared silently unless recorded.
    """
    return hashlib.sha1("\n".join(sorted(stems)).encode("utf-8")).hexdigest()


# ── step 1: the screening subsample ──────────────────────────────────────────
def video_stratified_fill(pool_by_video, already, target, cap, rng):
    """Draw ``target`` more images, proportional to video size, capped per video.

    Videos are picked by largest shortfall against their proportional share, so the
    draw tracks the split's own video mix instead of a flat per-video quota -- the
    test videos differ by 5x in frame count. ``cap`` stops any one video dominating.

    Args:
        pool_by_video: ``{video_id: [image_name, ...]}`` of *unselected* images.
        already: ``{video_id: n}`` already taken by the class fill.
        target: How many more images to draw.
        cap: Hard per-video ceiling on the final count.
        rng: Seeded ``random.Random``.

    Returns:
        The newly drawn image names.
    """
    remaining = {v: list(names) for v, names in pool_by_video.items() if names}
    for names in remaining.values():
        rng.shuffle(names)

    total = sum(len(names) for names in pool_by_video.values()) + sum(already.values())
    counts = dict(already)
    drawn = []
    while len(drawn) < target:
        best, best_key = None, None
        for video, names in remaining.items():
            if not names:
                continue
            taken = counts.get(video, 0)
            if taken >= cap:
                continue
            share = (len(names) + taken) / total if total else 0.0
            # Smallest realised-vs-proportional ratio first; ties by video id so the
            # draw is reproducible independently of dict ordering.
            key = (taken / share if share else float("inf"), video)
            if best_key is None or key < best_key:
                best, best_key = video, key
        if best is None:
            break  # every video is capped or exhausted
        drawn.append(remaining[best].pop())
        counts[best] = counts.get(best, 0) + 1
    return drawn


def make_subsample(args):
    """Draw and pin the screening subsample."""
    out_path = os.path.join(args.root, SUBSAMPLE_NAME)
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            "{} already exists. It is written once and then left alone so every "
            "variant screens the same images -- pass --force to redraw it, and "
            "expect every existing run to become incomparable.".format(out_path))

    names = read_prompt_classes(args.root)
    boxes = load_boxes(args.root, args.split)
    image_classes = classes_of(boxes)
    rng = random.Random(args.seed)

    # Stage 1 -- rarest-first class fill. Six test classes have fewer than six
    # images; without this the tail lands at zero support and per-class adherence
    # is unreadable exactly where the collapse is most interesting.
    covered, _ = select_images(image_classes, len(names), args.per_class_min, args.seed)
    selected = list(covered)
    if len(selected) > args.n:
        raise SystemExit(
            "the class fill alone took {} images, more than --n {}. Lower "
            "--per-class-min or raise --n.".format(len(selected), args.n))

    # Stage 2 -- video-stratified fill to --n.
    by_video = defaultdict(list)
    for name in sorted(boxes):
        if name not in covered:
            by_video[video_id_of(name)].append(name)
    already = Counter(video_id_of(name) for name in selected)
    cap = max(1, int(args.max_video_frac * args.n))
    selected += video_stratified_fill(by_video, already, args.n - len(selected), cap, rng)

    if len(selected) < args.n:
        print("[warning] only {} images available under the {}-per-video cap, "
              "wanted {}".format(len(selected), cap, args.n))

    stems = sorted(os.path.splitext(name)[0] for name in selected)
    video_support = Counter(video_id_of(name) for name in selected)
    record = {
        "seed": args.seed,
        "n": len(stems),
        "split": args.split,
        "per_class_min": args.per_class_min,
        "max_video_frac": args.max_video_frac,
        "sha1": fingerprint(stems),
        "stems": stems,
        "class_support": support_histograms(boxes, names, selected),
        "natural_class_support": support_histograms(boxes, names),
        "video_support": dict(sorted(video_support.items())),
        "split_images": len(boxes),
        "split_videos": len({video_id_of(name) for name in boxes}),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    print("wrote {}".format(out_path))
    print("  {} images from {} of {} videos, sha1 {}".format(
        record["n"], len(video_support), record["split_videos"], record["sha1"]))
    empty = [n for n, s in record["class_support"].items() if s["images"] == 0]
    print("  classes with zero support: {}".format(", ".join(empty) if empty else "none"))
    worst_video, worst_count = video_support.most_common(1)[0]
    print("  largest video {}: {} images ({:.1f}% of the subsample, cap {})".format(
        worst_video, worst_count, 100.0 * worst_count / record["n"], cap))
    print("\n  {:26s} {:>12s} {:>12s}".format("class", "subsample", "full split"))
    for name in names:
        sub, full = record["class_support"][name], record["natural_class_support"][name]
        print("  {:26s} {:5d} img {:3d} bx {:5d} img {:4d} bx".format(
            name, sub["images"], sub["boxes"], full["images"], full["boxes"]))
    return 0


def load_subsample(root):
    """Read the pinned subsample, failing loudly if it was never drawn."""
    path = os.path.join(root, SUBSAMPLE_NAME)
    if not os.path.isfile(path):
        raise SystemExit("{} not found -- run --make-subsample first".format(path))
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if fingerprint(record["stems"]) != record["sha1"]:
        raise SystemExit("{}: stems do not match the recorded sha1".format(path))
    return record


# ── step 2: the few-shot exemplar sets ───────────────────────────────────────
def greedy_class_cover(boxes, n_classes, k, max_boxes):
    """Order train frames by how many new prompt classes each one adds.

    Frames with more than ``max_boxes`` boxes are filtered out of the pool rather
    than truncated: dropping a box from a demonstration would show the model a
    visible animal that is *not* in the answer, which is exactly the "there is
    nothing more here" signal the training set never contained.

    Ties break toward fewer boxes (cleaner demonstrations), then toward a video not
    already used (so k=8 is not eight frames of one seafloor), then by name.

    Returns:
        ``[image_name, ...]`` of length ``k``, an ordering whose prefixes are the
        nested k=2/4/8 sets.
    """
    pool = {name: bs for name, bs in boxes.items() if 0 < len(bs) <= max_boxes}
    chosen, covered, used_videos = [], set(), set()
    while len(chosen) < k and pool:
        best, best_key = None, None
        for name in sorted(pool):
            present = {box[0] for box in pool[name]}
            key = (-len(present - covered), len(pool[name]),
                   video_id_of(name) in used_videos, name)
            if best_key is None or key < best_key:
                best, best_key = name, key
        chosen.append(best)
        covered |= {box[0] for box in pool[best]}
        used_videos.add(video_id_of(best))
        del pool[best]
    return chosen, covered


def denormalise(box, width, height):
    """YOLO ``cx cy w h`` in [0,1] -> ``[x1, y1, x2, y2]`` in original pixels."""
    _, cx, cy, bw, bh = box
    return [int(round((cx - bw / 2) * width)), int(round((cy - bh / 2) * height)),
            int(round((cx + bw / 2) * width)), int(round((cy + bh / 2) * height))]


def make_exemplars(args):
    """Pick the exemplar frames and write the nested exemplars_k<K>.json files."""
    from PIL import Image  # host-safe: no torch, and PIL is present in both places

    names = read_prompt_classes(args.root)
    boxes = load_boxes(args.root, "train")
    k_values = sorted(int(k) for k in args.exemplar_ks.split(","))
    chosen, covered = greedy_class_cover(boxes, len(names), max(k_values),
                                         args.exemplar_max_boxes)
    if len(chosen) < max(k_values):
        raise SystemExit("only {} train frames have 1..{} boxes; cannot build k={}".format(
            len(chosen), args.exemplar_max_boxes, max(k_values)))

    images_dir = os.path.join(args.root, "train", "images")
    entries = []
    for image_name in chosen:
        path = os.path.join(images_dir, image_name)
        with Image.open(path) as image:
            width, height = image.size
        entries.append({
            "image": path,
            "stem": os.path.splitext(image_name)[0],
            "video": video_id_of(image_name),
            "original_width": width,
            "original_height": height,
            "boxes": [{"bbox_2d": denormalise(box, width, height), "label": names[box[0]]}
                      for box in boxes[image_name]],
        })

    print("greedy cover: {} of {} prompt classes in {} frames".format(
        len(covered), len(names), len(chosen)))
    for i, entry in enumerate(entries, start=1):
        print("  {}. {} [video {}] {}".format(
            i, entry["stem"], entry["video"],
            ", ".join(box["label"] for box in entry["boxes"])))

    for k in k_values:
        out_path = os.path.join(args.root, "exemplars_k{}.json".format(k))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(entries[:k], handle, indent=2)
        classes_here = {box["label"] for entry in entries[:k] for box in entry["boxes"]}
        print("wrote {} -- {} frames, {} distinct classes".format(
            out_path, k, len(classes_here)))
    return 0



# ── step 1b: the free baseline row ───────────────────────────────────────────
def recover_baseline(args):
    """Carve the subsample out of the finished full-split zero-shot run.

    ``runs/thuenen_zeroshot`` already covers all 1303 test stems with the P0 prompt,
    so the subsample's P0 row costs no GPU time at all. Two things it can and cannot
    do:

    * It **can** check that the subsample is representative -- score it and the
      class-agnostic mAP should land near the full-split 0.0108 -- and it can serve
      as the correctness gate for the new runner, which must reproduce it.
    * It **cannot** supply the adherence metric. ``run_zero_shot`` wrote only
      ``snap_labels``-rewritten text, so there is no ``results_raw/``. Snapped text
      bounds the in-vocabulary share from above (out-of-vocab words that survived
      snapping are real, but a label reading as a prompt name may have been snapped
      there from a paraphrase), which is why ``P0_baseline`` is re-run fresh.
    """
    record = load_subsample(args.root)
    stems = set(record["stems"])

    src_results = os.path.join(args.source_run, "results")
    with open(os.path.join(args.source_run, "metadata.json"), encoding="utf-8") as handle:
        source_meta = json.load(handle)

    out_dir = os.path.join(args.runs, args.recovered_name)
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    linked, missing = 0, []
    for stem in sorted(stems):
        source = os.path.join(src_results, stem + ".txt")
        if not os.path.isfile(source):
            missing.append(stem)
            continue
        destination = os.path.join(results_dir, stem + ".txt")
        if os.path.islink(destination) or os.path.exists(destination):
            os.remove(destination)
        # Relative, so the link resolves from the host too and not only from the
        # container whose /workspace paths built it.
        os.symlink(os.path.relpath(source, results_dir), destination)
        linked += 1

    dims = {name: value for name, value in source_meta.get("image_dims", {}).items()
            if os.path.splitext(name)[0] in stems}
    metadata = {
        "prompt": source_meta.get("prompt"),
        "checkpoint": source_meta.get("checkpoint"),
        "image_dims": dims,
        "variant": args.recovered_name,
        "subsample_sha1": record["sha1"],
        "recovered_from": args.source_run,
        "note": "symlinked subset of a finished full-split run; snapped text only, "
                "no results_raw/ -- adherence numbers from it are an upper bound",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("linked {} of {} stems into {}".format(linked, len(stems), results_dir))
    if missing:
        print("[warning] {} subsample stems had no result in {}: {}".format(
            len(missing), src_results, ", ".join(missing[:5])))
    print("  metadata.json carries {} image_dims entries".format(len(dims)))
    return 0



# ── step 3: the runner ───────────────────────────────────────────────────────
def annotator(cache_dir, colour=(255, 0, 0), width=6):
    """Return a callable that burns an exemplar's boxes in as red rectangles.

    Only ``F0_caption_control`` needs this -- it reproduces the earlier one-turn
    format, where the target was marked in the pixels instead of written as
    coordinates. Results are cached per stem so the two-pass rebuild does not redraw.
    """
    from PIL import Image, ImageDraw

    os.makedirs(cache_dir, exist_ok=True)
    cache = {}

    def annotate(exemplar):
        stem = exemplar["stem"]
        if stem not in cache:
            out_path = os.path.join(cache_dir, stem + ".jpg")
            if not os.path.isfile(out_path):
                with Image.open(exemplar["image"]) as image:
                    image = image.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    for box in exemplar["boxes"]:
                        draw.rectangle(box["bbox_2d"], outline=colour, width=width)
                    image.save(out_path, quality=95)
            cache[stem] = out_path
        return cache[stem]

    return annotate


def build_inputs(processor, messages):
    """``apply_chat_template`` -> ``process_vision_info`` -> ``processor``."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    return text, inputs


def run_messages(model, processor, spec, ctx, exemplars, query_path, max_new_tokens):
    """Generate for one query image, exemplar boxes rescaled per exemplar.

    Every image in a multi-image message is resized independently, and the model
    reads and writes coordinates in *model-input* pixel space. So an exemplar's
    demonstrated boxes have to be rescaled by that exemplar's **own**
    ``image_grid_thw`` row -- not the query's, and not left in original pixels. This
    is the step a naive implementation gets silently wrong, and nothing downstream
    would flag it: the demonstrations would simply point at the wrong places.

    Two passes, mirroring ``infer_fewshot.py:192-207``. The grids depend only on the
    images and the pixel budgets, never on the text, so the second pass is exact.

    Returns:
        ``(response_text, input_height, input_width, prompt_tokens, rendered_text)``
        where the dimensions are the **query's**, for ``metadata.json``.
    """
    import torch

    from batch_inference import image_token_id
    from infer_fewshot import double_image_tokens
    from prompt_variants import scale_exemplars

    ctx.exemplars = exemplars
    messages = spec["build"](ctx, query_path)
    text, inputs = build_inputs(processor, messages)

    grids = inputs["image_grid_thw"]
    if exemplars:
        # grids[:-1] are the exemplars in message order; grids[-1] is the query,
        # because every builder puts the query image last.
        scales = []
        for exemplar, grid in zip(exemplars, grids[:-1]):
            input_h, input_w = grid[1].item() * 14, grid[2].item() * 14
            scales.append((input_w / exemplar["original_width"],
                           input_h / exemplar["original_height"]))
        ctx.exemplars = scale_exemplars(exemplars, scales)
        messages = spec["build"](ctx, query_path)
        text, inputs = build_inputs(processor, messages)
        grids = inputs["image_grid_thw"]

    input_height = grids[-1][1].item() * 14
    input_width = grids[-1][2].item() * 14

    inputs["input_ids"], inputs["attention_mask"] = double_image_tokens(
        inputs, image_token_id)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out_ids[len(in_ids):]
               for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    return response, input_height, input_width, prompt_tokens, text


def query_path_for(images_dir, stem):
    """The query image for one subsample stem."""
    return os.path.join(images_dir, stem + ".jpg")


def run_variant(args, model, processor, name, stems, prompt_names, images_dir):
    """Run one variant over the subsample and write its run dir."""
    from tqdm import tqdm
    from PIL import Image

    from nautilus_zeroshot import snap_labels
    from prompt_variants import VARIANTS, Context

    spec = VARIANTS[name]
    record = load_subsample(args.root)

    exemplars = []
    if spec["exemplars"]:
        path = os.path.join(args.root, "exemplars_k{}.json".format(spec["exemplars"]))
        if not os.path.isfile(path):
            raise SystemExit("{} not found -- run --make-exemplars first".format(path))
        with open(path, encoding="utf-8") as handle:
            exemplars = json.load(handle)

    out_dir = os.path.join(args.runs, name)
    results_dir = os.path.join(out_dir, "results")
    raw_dir = os.path.join(out_dir, "results_raw")
    for directory in (results_dir, raw_dir):
        os.makedirs(directory, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.json")

    ctx = Context(prompt_names, args.query_max_pixels * 28 * 28,
                  args.exemplar_max_pixels * 28 * 28,
                  annotate=annotator(os.path.join(args.runs, "_annotated_exemplars"))
                  if spec.get("annotated") else None)

    # Merge into any existing metadata, the way batch_inference.load_metadata does.
    # Without this, --skip-existing would resume with an empty image_dims and every
    # already-finished image would lose the input_width/input_height that
    # evaluate_detections.py needs to rescale its boxes -- a resumed run would score
    # as if most of it had never happened.
    previous = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            previous = json.load(handle)

    metadata = {
        "prompt": previous.get("prompt"),  # else the first rendered query turn
        "checkpoint": args.checkpoint,
        "image_dims": previous.get("image_dims", {}),
        "variant": name,
        "note": spec.get("note"),
        "subsample_sha1": record["sha1"],
        "subsample_n": record["n"],
        "exemplars": [e["stem"] for e in exemplars],
        "exemplar_max_pixels": args.exemplar_max_pixels,
        "query_max_pixels": args.query_max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "snapped": spec.get("snap", True),
        "prompt_tokens": previous.get("prompt_tokens", {}),
    }
    stats = Counter()
    over_budget = []

    for stem in tqdm(stems, desc=name):
        raw_path = os.path.join(raw_dir, stem + ".txt")
        out_path = os.path.join(results_dir, stem + ".txt")
        if args.skip_existing and os.path.isfile(raw_path) and os.path.isfile(out_path):
            stats["skipped"] += 1
            if os.path.basename(query_path_for(images_dir, stem)) not in metadata["image_dims"]:
                print("[warning] {} was already written but has no metadata entry; "
                      "delete it to regenerate".format(stem))
            continue
        query_path = query_path_for(images_dir, stem)
        try:
            response, input_height, input_width, prompt_tokens, text = run_messages(
                model, processor, spec, ctx, exemplars, query_path, args.max_new_tokens)
        except Exception as error:  # noqa: BLE001
            print("[failed] {}: {}".format(stem, error))
            stats["failed"] += 1
            continue

        if metadata["prompt"] is None:
            metadata["prompt"] = text
            if args.dump_text:
                dump = os.path.join(out_dir, "rendered_prompt.txt")
                with open(dump, "w", encoding="utf-8") as handle:
                    handle.write(text)
                print("\nrendered prompt for {} -> {} ({} tokens)".format(
                    stem, dump, prompt_tokens))

        # model_max_length is 8192, but that is a *training-length* convention:
        # max_position_embeddings is 128000 and sliding_window 32768, so an overlong
        # prompt degrades the output rather than raising. Record it, never abort.
        if prompt_tokens > args.token_warn:
            over_budget.append((stem, prompt_tokens))

        with open(raw_path, "w", encoding="utf-8") as handle:
            handle.write(response)
        if spec.get("snap", True):
            snapped, unmatched = snap_labels(response, prompt_names)
            stats["unmatched_labels"] += unmatched
        else:
            snapped = response
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(snapped)

        with Image.open(query_path) as image:
            width, height = image.size
        metadata["image_dims"][os.path.basename(query_path)] = {
            "input_height": input_height,
            "input_width": input_width,
            "original_width": width,
            "original_height": height,
        }
        metadata["prompt_tokens"][stem] = prompt_tokens
        stats["ok"] += 1
        if stats["ok"] % 50 == 0:
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    tokens = list(metadata["prompt_tokens"].values())
    print("{}: ok={} skipped={} failed={} unmatched_labels={}".format(
        name, stats["ok"], stats["skipped"], stats["failed"],
        stats["unmatched_labels"]))
    if tokens:
        print("  prompt_tokens mean {:.0f} max {}".format(
            sum(tokens) / len(tokens), max(tokens)))
    if over_budget:
        print("  [warning] {} images over {} prompt tokens (max {}). Output may be "
              "degraded; lower --exemplar-max-pixels and re-run this variant."
              .format(len(over_budget), args.token_warn,
                      max(t for _, t in over_budget)))
    return stats


def run_variants(args):
    """Load the model once, then run every requested variant through it."""
    # Deferred so --make-subsample / --make-exemplars stay torch-free.
    sys.path.insert(0, os.path.dirname(SCRIPTS))
    import torch
    from transformers import AutoProcessor

    from prompt_variants import SWEEP, VARIANTS
    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_VL_Nautilus_ForConditionalGeneration,
    )

    requested = SWEEP if args.variants.strip() == "all" else [
        v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested if v not in VARIANTS]
    if unknown:
        raise SystemExit("unknown variant(s): {}\nknown: {}".format(
            ", ".join(unknown), ", ".join(sorted(VARIANTS))))

    record = load_subsample(args.root)
    stems = record["stems"][:args.limit] if args.limit else record["stems"]
    prompt_names = read_prompt_classes(args.root)
    images_dir = os.path.join(args.root, args.split, "images")
    print("{} variants x {} images on cuda:{}".format(
        len(requested), len(stems), args.device))

    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:" + args.device,
    )
    model.eval()
    # The global cap has to cover the query; the smaller exemplar budget is applied
    # per image through each message's own "max_pixels" key.
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=1 * 28 * 28,
        max_pixels=args.query_max_pixels * 28 * 28)

    for name in requested:
        run_variant(args, model, processor, name, stems, prompt_names, images_dir)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="The thuenen_scaling split root.")
    parser.add_argument("--runs", default=DEFAULT_RUNS,
                        help="Parent directory for the per-variant run dirs.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="Split the subsample is drawn from.")

    parser.add_argument("--make-subsample", action="store_true",
                        help="Draw the screening subsample and exit.")
    parser.add_argument("--n", type=int, default=250, help="Subsample size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class-min", type=int, default=3,
                        help="Images per prompt class the rarest-first fill targets.")
    parser.add_argument("--max-video-frac", type=float, default=0.15,
                        help="Hard ceiling on any one video's share of the subsample.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing screening_subsample.json.")

    parser.add_argument("--recover-baseline", action="store_true",
                        help="Symlink the subsample out of a finished full-split run.")
    parser.add_argument("--source-run", default="/workspace/runs/thuenen_zeroshot",
                        help="Finished run to carve the recovered baseline out of.")
    parser.add_argument("--recovered-name", default="P0_baseline_recovered",
                        help="Run-dir name for the recovered baseline.")

    parser.add_argument("--make-exemplars", action="store_true",
                        help="Pick the few-shot exemplar frames and exit.")
    parser.add_argument("--exemplar-ks", default="2,4,8",
                        help="Comma-separated exemplar counts; sets are nested.")
    parser.add_argument("--exemplar-max-boxes", type=int, default=4,
                        help="Frames with more boxes than this are not eligible.")

    parser.add_argument("--variants", default="",
                        help="Comma-separated variant names, or 'all' for the "
                             "whole sweep. See prompt_variants.SWEEP.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="0", help="CUDA device index.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--query-max-pixels", type=int, default=1338,
                        help="Query pixel budget in 28x28 patches; 1338 is the "
                             "training-time default.")
    parser.add_argument("--exemplar-max-pixels", type=int, default=256,
                        help="Per-exemplar pixel budget in 28x28 patches. Lower it "
                             "if k=8 runs over the token budget.")
    parser.add_argument("--token-warn", type=int, default=8192,
                        help="Warn above this prompt length (tokenizer_config's "
                             "model_max_length).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N subsample images (smoke test).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that already have both output files.")
    parser.add_argument("--dump-text", action="store_true",
                        help="Write the rendered chat template of the first query "
                             "to <run>/rendered_prompt.txt.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.make_subsample:
        return make_subsample(args)
    if args.make_exemplars:
        return make_exemplars(args)
    if args.recover_baseline:
        return recover_baseline(args)
    if args.variants:
        return run_variants(args)
    raise SystemExit("nothing to do: pass --make-subsample, --make-exemplars, "
                     "--recover-baseline or --variants")


if __name__ == "__main__":
    raise SystemExit(main())
