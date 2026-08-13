"""Prepare the Thünen scaling split and run NAUTILUS zero-shot detection on it.

This is one half of the "when does YOLO run out of data?" experiment. The other
half is ``yolo_scaling.py``, which trains YOLOv8 on progressively smaller slices
of the *same* split. Both are scored with ``evaluate_detections.py`` so the two
numbers are directly comparable.

Two modes
---------
``--prepare``
    Turn ``datasets/thuenen`` (one directory per class, frames duplicated across
    the classes annotated on them) into a single multi-class dataset split
    train/val/test **by video**, and write it to ``--out``. Needs no GPU and no
    torch. Frames are hardlinked, so the split costs almost no disk.

    Videos, not frames, are the split unit: consecutive frames of one video are
    near-duplicates, and a frame-level split would let a 1-sample YOLO model
    score well by recognising the neighbouring frame it trained on.

default (inference)
    Run NAUTILUS zero-shot over ``<out>/test/images`` and write per-image
    results in the same layout ``batch_inference.py`` produces
    (``results/<stem>.txt`` + ``metadata.json``).

Label spaces
------------
The split carries two sets of label files:

``labels/`` + ``classes.txt``
    The 29 fine-grained BIIGLE classes. What YOLO trains on.

``labels_prompt/`` + ``classes_prompt.txt``
    The same boxes re-indexed onto the English prompt names in
    ``prompt_names.csv``. Several BIIGLE classes collapse here (both *Ophiura*
    classes are "brittle star"; *Pagurus bernhardus* and *Paguridae* are both
    "hermit crab") because no zero-shot prompt can separate them. This is the
    space the NAUTILUS-vs-YOLO head-to-head is scored in, for both models.

``prompt_names.csv`` is written once with the defaults below and then left
alone -- edit it and re-run ``--prepare`` to change the prompts.

Usage:
    # 1. build the split (host or container, no GPU needed)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/nautilus_zeroshot.py --prepare

    # 2. hardlink the split where the brackish container can see it
    cp -al /home/tfricke/nautilus/datasets/thuenen_scaling \
           /home/tfricke/brackish/container/thuenen_scaling

    # 3. zero-shot inference on the test split
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/nautilus_zeroshot.py --checkpoint /workspace/weights/qwen-instruct-7b-weights

    # 4. score it
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_zeroshot/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_zeroshot/metrics.json
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

DEFAULT_DATASET = "/workspace/datasets/thuenen"
DEFAULT_OUT = "/workspace/datasets/thuenen_scaling"
DEFAULT_RUN_DIR = "/workspace/runs/thuenen_zeroshot"
DEFAULT_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Frames are named "<video stem>(<video id>)_f<frame index>.jpg" by build_dataset.py.
VIDEO_ID_PATTERN = re.compile(r"\((\d+)\)_f\d+$")

SPLITS = ("train", "val", "test")

# English names NAUTILUS has a chance of knowing, per BIIGLE label. Written to
# prompt_names.csv on the first --prepare run; edit that file, not this table.
DEFAULT_PROMPT_NAMES = {
    "Ophiura_Schlangenstern": "brittle star",
    "Asterias rubens_gemeiner Seestern": "starfish",
    "Ophiura ophiura_Großer Schlangenstern": "brittle star",
    "Turitella communis": "tower shell snail",
    "Muscheln_Bivalvia": "bivalve shell",
    "Liocarcinus_Schwimmkrabben": "swimming crab",
    "Rippenquallen_Ctenophora": "comb jelly",
    "Blumentiere_Anthozoa": "sea anemone",
    "Unidentified organism": "unidentified organism",
    "Corystes cassivelaunus_Antennenkrebs": "masked crab",
    "Ammodytes_Sandaale": "sand eel",
    "Astropecten irregularis_nordischer Kammstern": "sand star",
    "Pleuronectiformes_Plattfische": "flatfish",
    "Callionymus_Leierfische": "dragonet",
    "Pagurus bernhardus_gemeiner Einsiedlerkrebs": "hermit crab",
    "Fische_Teleostei": "fish",
    "Tierspuren": "animal track",
    "Liocarcinus holsatus_glatte Schwimmkrabbe": "swimming crab",
    "Paguridae_Einsiedlerkrebse": "hermit crab",
    "Syngnathus rostellatus_kleine Seenadel": "pipefish",
    "Pomatoschistus_Grundeln": "goby",
    "Stachelhäuter_Echinodermata": "echinoderm",
    "Caridea_Garnelen": "shrimp",
    "Ensis ensis_ Schwertmuschel": "razor clam",
    "Mactra spp.": "trough shell clam",
    "Pleuronectidae": "flatfish",
    "Sngnathus_spec": "pipefish",
    "Echinoidea_Seeigel": "sea urchin",
    "Agonus cataphractus_Steinpicker": "hooknose fish",
}

PROMPT_TEMPLATE = (
    "Detect all objects in the image and output their locations as bounding "
    "boxes with labels. Possible Objects are {classes}"
)


# ── split preparation ────────────────────────────────────────────────────────
def video_id_of(image_name):
    """Return the BIIGLE video id embedded in a frame filename."""
    match = VIDEO_ID_PATTERN.search(os.path.splitext(image_name)[0])
    if match is None:
        raise ValueError("cannot read a video id from {!r}".format(image_name))
    return match.group(1)


def read_class_map(dataset_dir, min_annotations):
    """Return the kept classes, in the dataset's own frequency order.

    Args:
        dataset_dir: Root written by ``build_dataset.py``.
        min_annotations: Drop classes with fewer parsed annotations than this.

    Returns:
        List of dicts with ``old_id``, ``new_id``, ``label_name``, ``dir_name``.
    """
    path = os.path.join(dataset_dir, "class_map.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    kept = []
    for row in rows:
        if int(row["annotations_parsed"]) < min_annotations:
            continue
        kept.append({
            "old_id": int(row["class_id"]),
            "new_id": len(kept),
            "label_name": row["label_name"],
            "dir_name": row["dir_name"],
        })
    return kept


def collect_boxes(dataset_dir, classes):
    """Merge the per-class label files back into one multi-class label per frame.

    Args:
        dataset_dir: Root written by ``build_dataset.py``.
        classes: Output of :func:`read_class_map`.

    Returns:
        ``(boxes, image_paths)``. ``boxes`` maps an image filename to a list of
        ``(new_class_id, cx, cy, w, h)``; ``image_paths`` maps the same filename
        to one source path on disk (the frame is byte-identical in every class
        directory it appears in).
    """
    boxes = defaultdict(list)
    image_paths = {}

    for entry in classes:
        class_root = os.path.join(dataset_dir, entry["dir_name"])
        images_dir = os.path.join(class_root, "images")
        labels_dir = os.path.join(class_root, "labels")
        if not os.path.isdir(images_dir):
            print("[warning] no images/ under {}".format(class_root), file=sys.stderr)
            continue

        for image_name in sorted(os.listdir(images_dir)):
            if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            image_paths.setdefault(image_name, os.path.join(images_dir, image_name))

            label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
            if not os.path.isfile(label_path):
                continue
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    if int(float(parts[0])) != entry["old_id"]:
                        # A frame lives in several class directories, but each
                        # copy of its label file only lists that class's boxes.
                        continue
                    boxes[image_name].append(
                        (entry["new_id"],) + tuple(float(v) for v in parts[1:5]))

    return boxes, image_paths


# Some classes sit almost entirely in one video (Ophiura ophiura: 52% of its
# frames in a single video), so an exact 60/20/20 is unreachable for them. The
# penalty is therefore asymmetric: a split that is *short* of its target hurts,
# a train split that overshoots does not, and an oversized val/test hurts only a
# little. A symmetric cost prefers dumping such a video into val, which is how
# the first version of this produced a 53%-val class.
OVERSHOOT_WEIGHT = 0.25
EMPTY_PENALTY = 1.0


def split_cost(realized, targets, totals):
    """Per-class deviation from the target split, plus a penalty for empty splits."""
    cost = 0.0
    for class_id, total in totals.items():
        if total <= 0:
            continue
        for split in SPLITS:
            deviation = (realized[split][class_id] - targets[split][class_id]) / total
            if deviation < 0.0:
                cost += deviation ** 2
            elif split != "train":
                cost += OVERSHOOT_WEIGHT * deviation ** 2
            if realized[split][class_id] == 0:
                cost += EMPTY_PENALTY
    return cost


def assign_videos(video_counts, class_ids, val_frac, test_frac, seed, restarts, iterations):
    """Assign whole videos to train/val/test so per-class counts hit the target ratio.

    Randomised greedy (videos placed largest-first into the split that costs
    least) followed by hill-climbing over single-video moves and pairwise swaps.
    Exact balance is not reachable -- some classes sit in only a handful of
    videos -- so this minimises :func:`split_cost` instead.

    Args:
        video_counts: ``{video_id: {class_id: n_images}}``.
        class_ids: All class ids that must be balanced.
        val_frac: Target fraction of each class's images in val.
        test_frac: Target fraction of each class's images in test.
        seed: RNG seed.
        restarts: Independent randomised greedy runs; the cheapest is kept.
        iterations: Hill-climbing moves attempted per restart.

    Returns:
        ``{video_id: split}``.
    """
    videos = sorted(video_counts)
    totals = {c: sum(video_counts[v].get(c, 0) for v in videos) for c in class_ids}
    targets = {
        "train": {c: totals[c] * (1.0 - val_frac - test_frac) for c in class_ids},
        "val": {c: totals[c] * val_frac for c in class_ids},
        "test": {c: totals[c] * test_frac for c in class_ids},
    }

    def move(realized, video, source, destination):
        realized[source].subtract(video_counts[video])
        realized[destination].update(video_counts[video])

    best_assignment, best_cost = None, float("inf")
    for restart in range(restarts):
        rng = random.Random(seed + restart)
        order = sorted(videos, key=lambda v: (-sum(video_counts[v].values()), v))
        # Jitter the order so restarts explore different greedy paths.
        for index in range(len(order) - 1, 0, -1):
            if rng.random() < 0.3:
                swap = rng.randrange(index + 1)
                order[index], order[swap] = order[swap], order[index]

        assignment = {}
        realized = {s: Counter() for s in SPLITS}
        for video in order:
            best_split, best_local = None, float("inf")
            for split in SPLITS:
                realized[split].update(video_counts[video])
                local = split_cost(realized, targets, totals)
                realized[split].subtract(video_counts[video])
                if local < best_local:
                    best_split, best_local = split, local
            assignment[video] = best_split
            realized[best_split].update(video_counts[video])

        cost = split_cost(realized, targets, totals)
        for step in range(iterations):
            if step % 2 == 0:
                # Single move: send one video to a different split.
                video = rng.choice(videos)
                source = assignment[video]
                destination = rng.choice([s for s in SPLITS if s != source])
                move(realized, video, source, destination)
                candidate_cost = split_cost(realized, targets, totals)
                if candidate_cost < cost:
                    assignment[video] = destination
                    cost = candidate_cost
                else:
                    move(realized, video, destination, source)
            else:
                # Swap: exchange two videos between their splits. Escapes the
                # local optima that single moves cannot, because relocating one
                # dominant video alone always looks like a step backwards.
                first, second = rng.sample(videos, 2)
                if assignment[first] == assignment[second]:
                    continue
                a, b = assignment[first], assignment[second]
                move(realized, first, a, b)
                move(realized, second, b, a)
                candidate_cost = split_cost(realized, targets, totals)
                if candidate_cost < cost:
                    assignment[first], assignment[second] = b, a
                    cost = candidate_cost
                else:
                    move(realized, first, b, a)
                    move(realized, second, a, b)

        if cost < best_cost:
            best_assignment, best_cost = dict(assignment), cost

    print("split cost {:.4f} over {} restarts".format(best_cost, restarts))
    return best_assignment


def link_or_copy(source, destination):
    """Hardlink ``source`` to ``destination``, falling back to a copy across devices."""
    if os.path.exists(destination):
        os.remove(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def write_split(out_dir, split, image_names, image_paths, boxes, prompt_id_by_class):
    """Materialise one split: hardlinked frames plus both label spaces."""
    images_dir = os.path.join(out_dir, split, "images")
    labels_dir = os.path.join(out_dir, split, "labels")
    prompt_dir = os.path.join(out_dir, split, "labels_prompt")
    for directory in (images_dir, labels_dir, prompt_dir):
        # Wiped, not merged: re-running --prepare draws a different assignment,
        # and frames left behind by the previous one would leak across splits.
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    for image_name in image_names:
        link_or_copy(image_paths[image_name], os.path.join(images_dir, image_name))
        stem = os.path.splitext(image_name)[0]
        fine, prompt = [], []
        for class_id, cx, cy, bw, bh in boxes[image_name]:
            geometry = "{:.6f} {:.6f} {:.6f} {:.6f}\n".format(cx, cy, bw, bh)
            fine.append("{} {}".format(class_id, geometry))
            prompt.append("{} {}".format(prompt_id_by_class[class_id], geometry))
        with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("".join(fine))
        with open(os.path.join(prompt_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("".join(prompt))


def load_prompt_names(out_dir, classes):
    """Read prompt_names.csv, creating it from the built-in table when absent."""
    path = os.path.join(out_dir, "prompt_names.csv")
    if os.path.isfile(path):
        with open(path, encoding="utf-8", newline="") as handle:
            mapping = {row["label_name"]: row["prompt_name"].strip()
                       for row in csv.DictReader(handle)}
        missing = [c["label_name"] for c in classes if not mapping.get(c["label_name"])]
        if missing:
            raise SystemExit(
                "{} has no prompt_name for: {}".format(path, ", ".join(missing)))
        print("using prompt names from {}".format(path))
        return mapping

    mapping = {}
    for entry in classes:
        name = DEFAULT_PROMPT_NAMES.get(entry["label_name"])
        if not name:
            raise SystemExit(
                "no default prompt name for {!r} -- add it to DEFAULT_PROMPT_NAMES "
                "or write {} by hand".format(entry["label_name"], path))
        mapping[entry["label_name"]] = name

    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "label_name", "prompt_name"])
        for entry in classes:
            writer.writerow([entry["new_id"], entry["label_name"], mapping[entry["label_name"]]])
    print("wrote {} -- edit it and re-run --prepare to change the prompts".format(path))
    return mapping


def prepare(args):
    """Build the video-grouped multi-class split under ``args.out``."""
    classes = read_class_map(args.dataset, args.min_annotations)
    print("{} classes with >= {} annotations".format(len(classes), args.min_annotations))

    prompt_by_label = load_prompt_names(args.out, classes)
    prompt_names = []
    prompt_id_by_class = {}
    for entry in classes:
        name = prompt_by_label[entry["label_name"]]
        if name not in prompt_names:
            prompt_names.append(name)
        prompt_id_by_class[entry["new_id"]] = prompt_names.index(name)
    merged = len(classes) - len(prompt_names)
    print("{} fine classes -> {} prompt classes ({} merged as synonyms)".format(
        len(classes), len(prompt_names), merged))

    boxes, image_paths = collect_boxes(args.dataset, classes)
    images = sorted(image_paths)
    print("{} unique frames, {} boxes".format(
        len(images), sum(len(v) for v in boxes.values())))

    images_by_video = defaultdict(list)
    for image_name in images:
        images_by_video[video_id_of(image_name)].append(image_name)

    class_ids = [entry["new_id"] for entry in classes]
    video_counts = {}
    for video, names in images_by_video.items():
        counter = Counter()
        for image_name in names:
            for class_id in {b[0] for b in boxes[image_name]}:
                counter[class_id] += 1
        video_counts[video] = counter

    assignment = assign_videos(video_counts, class_ids, args.val_frac, args.test_frac,
                               args.seed, args.restarts, args.iterations)

    split_images = {s: [] for s in SPLITS}
    for video, names in sorted(images_by_video.items()):
        split_images[assignment[video]].extend(sorted(names))

    for split in SPLITS:
        write_split(args.out, split, split_images[split], image_paths, boxes,
                    prompt_id_by_class)
        print("{:>5}: {} videos, {} frames".format(
            split, sum(1 for v in assignment.values() if v == split),
            len(split_images[split])))

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as handle:
        for entry in classes:
            handle.write(entry["label_name"] + "\n")
    with open(os.path.join(args.out, "classes_prompt.txt"), "w", encoding="utf-8") as handle:
        for name in prompt_names:
            handle.write(name + "\n")

    # Per-class image counts, and a warning for classes a video-grouped split
    # cannot spread properly.
    per_class = {}
    for entry in classes:
        counts = {}
        for split in SPLITS:
            counts[split] = sum(
                1 for name in split_images[split]
                if any(b[0] == entry["new_id"] for b in boxes[name]))
        counts["videos"] = sum(1 for v, c in video_counts.items() if c.get(entry["new_id"]))
        per_class[entry["label_name"]] = counts
        empty = [s for s in SPLITS if counts[s] == 0]
        if empty:
            print("[warning] {!r} has no images in {} -- it occurs in only {} video(s), "
                  "so a video-grouped split cannot spread it. Consider excluding it "
                  "from the comparison.".format(
                      entry["label_name"], "/".join(empty), counts["videos"]))

    manifest = {
        "dataset": args.dataset,
        "min_annotations": args.min_annotations,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "classes": classes,
        "prompt_names": prompt_names,
        "prompt_id_by_class": prompt_id_by_class,
        "videos": {s: sorted(v for v, a in assignment.items() if a == s) for s in SPLITS},
        "images": split_images,
        "per_class": per_class,
    }
    with open(os.path.join(args.out, "splits.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print("\nwrote {}".format(os.path.join(args.out, "splits.json")))
    print("hardlink it into the brackish tree with:\n  cp -al {} {}".format(
        args.out.replace("/workspace/datasets", "/home/tfricke/nautilus/datasets"),
        "/home/tfricke/brackish/container/" + os.path.basename(args.out)))
    return 0


# ── zero-shot inference ──────────────────────────────────────────────────────
def canonicalise(text):
    """Lowercase and collapse whitespace/punctuation so label matching is forgiving."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def snap_labels(response, prompt_names):
    """Rewrite predicted labels onto the closest prompt name.

    The model paraphrases ("a starfish", "Starfish"); the evaluator compares
    label strings exactly, so anything that does not resolve to one of the
    prompt names would silently count as a wrong class.

    Args:
        response: Raw model output text.
        prompt_names: The prompt-space class names.

    Returns:
        ``(rewritten_text, n_unmatched)``.
    """
    lookup = {canonicalise(name): name for name in prompt_names}
    unmatched = [0]

    def replace(match):
        raw = match.group(1)
        key = canonicalise(raw)
        if key in lookup:
            return '"label": "{}"'.format(lookup[key])
        hits = [name for canonical, name in lookup.items()
                if canonical and (canonical in key or key in canonical)]
        if len(set(hits)) == 1:
            return '"label": "{}"'.format(hits[0])
        unmatched[0] += 1
        return match.group(0)

    rewritten = re.sub(r'"label"\s*:\s*"([^"]*)"', replace, response)
    return rewritten, unmatched[0]


def run_zero_shot(args):
    """Run NAUTILUS over the test split and write batch_inference-style results."""
    # Imported here so --prepare stays torch-free.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch  # noqa: F401  (imported for its side effects on CUDA init)
    from PIL import Image
    from transformers import AutoProcessor
    from tqdm import tqdm

    from batch_inference import find_images, run_inference
    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_VL_Nautilus_ForConditionalGeneration,
    )

    with open(os.path.join(args.out, "classes_prompt.txt"), encoding="utf-8") as handle:
        prompt_names = [line.strip() for line in handle if line.strip()]

    prompt = args.prompt or PROMPT_TEMPLATE.format(classes=", ".join(prompt_names))
    print("prompt: {}".format(prompt))

    image_dir = os.path.join(args.out, args.split, "images")
    images = find_images(__import__("pathlib").Path(image_dir), recursive=False)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("no images under {}".format(image_dir))
    print("{} query images from {}".format(len(images), image_dir))

    results_dir = os.path.join(args.run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    metadata_path = os.path.join(args.run_dir, "metadata.json")

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

    metadata = {"prompt": prompt, "checkpoint": args.checkpoint, "image_dims": {}}
    stats = Counter()
    for image_path in tqdm(images, desc="NAUTILUS zero-shot"):
        out_path = os.path.join(results_dir, image_path.stem + ".txt")
        if args.skip_existing and os.path.exists(out_path):
            stats["skipped"] += 1
            continue
        try:
            response, input_height, input_width = run_inference(
                model, processor, image_path, prompt, args.max_new_tokens)
        except Exception as error:  # noqa: BLE001
            print("[failed] {}: {}".format(image_path, error))
            stats["failed"] += 1
            continue

        snapped, unmatched = snap_labels(response, prompt_names)
        stats["unmatched_labels"] += unmatched
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
        stats["ok"] += 1
        if stats["ok"] % 50 == 0:
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nok={ok} skipped={skipped} failed={failed}, "
          "{unmatched_labels} predicted labels did not map onto a prompt name".format(
              ok=stats["ok"], skipped=stats["skipped"], failed=stats["failed"],
              unmatched_labels=stats["unmatched_labels"]))
    print("results: {}\nmetadata: {}".format(results_dir, metadata_path))
    print("\nscore it with:\n"
          "  python {}/evaluate_detections.py \\\n"
          "    --gt-dir {}/{}/labels_prompt \\\n"
          "    --pred-dir {} \\\n"
          "    --image-dir {} \\\n"
          "    --classes-file {}/classes_prompt.txt \\\n"
          "    --save-json {}/metrics.json".format(
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              args.out, args.split, results_dir, image_dir, args.out, args.run_dir))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true",
                        help="Build the split instead of running inference.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Per-class dataset written by build_dataset.py.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Where the merged, split dataset lives.")
    parser.add_argument("--min-annotations", type=int, default=10,
                        help="Drop classes with fewer parsed annotations than this.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=40,
                        help="Randomised greedy restarts for the video assignment.")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="Hill-climbing moves per restart.")

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                        help="Where results/ and metadata.json are written.")
    parser.add_argument("--split", default="test", choices=SPLITS,
                        help="Which split to run inference on.")
    parser.add_argument("--prompt", default=None,
                        help="Override the generated all-classes prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1338,
                        help="Query pixel budget in 28x28 patches. 1338 is the "
                             "training-time default; lower it if the 17.5 GB of "
                             "weights plus activations will not fit on the GPU.")
    parser.add_argument("--device", default="0", help="CUDA device index.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N images (smoke test).")
    args = parser.parse_args()

    return prepare(args) if args.prepare else run_zero_shot(args)


if __name__ == "__main__":
    sys.exit(main())
