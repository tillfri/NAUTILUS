"""Train YOLOv8 on the Thünen split at shrinking training-set sizes.

The other half of the "when does YOLO run out of data?" experiment; the split
itself is built by ``nautilus_zeroshot.py --prepare``. This script trains one
multi-class YOLOv8 model per training budget

    1, 2, 4, 8, 16, 32, 64, 128, full   images per class

against a val and test split that stay **fixed across every budget** -- only the
training slice changes -- and reports where the curve falls off. NAUTILUS needs
none of these images, so its zero-shot score is a flat line the curve is read
against.

Each run writes two kinds of numbers:

* Ultralytics' own ``mAP@0.5`` / ``mAP@0.5:0.95`` over the 29 fine classes, from
  ``model.val(split="test")``. This is the scaling curve.
* Test-set predictions re-emitted in the layout ``batch_inference.py`` produces
  (``nautilus_format/results/<stem>.txt`` + ``metadata.json``), labelled with the
  English prompt names. Feeding those to ``evaluate_detections.py`` scores YOLO
  with the exact same code as NAUTILUS, which is the only way the two are
  comparable -- Ultralytics integrates a confidence-ranked PR curve, while
  NAUTILUS emits no scores and is measured at a single operating point.

Epoch budget: a fixed epoch count is unfair to the tiny budgets (29 training
images at batch 16 is 2 optimiser steps per epoch). ``--min-updates`` raises the
epoch count for small budgets so every run gets a comparable number of gradient
steps, capped by ``--max-epochs``.

Usage:
    # inside the brackish container, after
    #   cp -al /home/tfricke/nautilus/datasets/thuenen_scaling \
    #          /home/tfricke/brackish/container/thuenen_scaling
    docker exec brackish python /usr/src/ultralytics/brackish/yolo_scaling.py \
        --root /usr/src/ultralytics/brackish/thuenen_scaling \
        --weights yolov8m.pt --device 0

    # shard the sweep over the four GPUs
    docker exec brackish python .../yolo_scaling.py --budgets 1,2 --device 0 &
    docker exec brackish python .../yolo_scaling.py --budgets 4,8 --device 1 &
    ...

To score a run against NAUTILUS, the exported predictions have to reach the
nautilus-qwen container, which does not mount the brackish tree. They are a few
MB of text:

    cp -r /home/tfricke/brackish/container/thuenen_scaling_runs/n8_s0/nautilus_format \
          /home/tfricke/nautilus/runs/thuenen_yolo_n8_s0
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_yolo_n8_s0/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_yolo_n8_s0/metrics.json
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

DEFAULT_ROOT = "/usr/src/ultralytics/brackish/thuenen_scaling"
DEFAULT_PROJECT = "/usr/src/ultralytics/brackish/thuenen_scaling_runs"
DEFAULT_BUDGETS = "1,2,4,8,16,32,64,128,full"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(root):
    """Read classes.txt, prompt_names.csv and the train-split labels.

    Returns:
        ``(class_names, prompt_by_class_id, image_classes)`` where
        ``image_classes`` maps a train image filename to the set of class ids
        annotated on it.
    """
    with open(os.path.join(root, "classes.txt"), encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]

    prompt_by_class_id = {}
    with open(os.path.join(root, "prompt_names.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_by_class_id[int(row["class_id"])] = row["prompt_name"].strip()

    images_dir = os.path.join(root, "train", "images")
    labels_dir = os.path.join(root, "train", "labels")
    image_classes = {}
    for image_name in sorted(os.listdir(images_dir)):
        if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        found = set()
        if os.path.isfile(label_path):
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if parts:
                        found.add(int(float(parts[0])))
        image_classes[image_name] = found

    return class_names, prompt_by_class_id, image_classes


def select_images(image_classes, n_classes, budget, seed):
    """Pick training images so every class reaches ``budget`` images where possible.

    Classes are filled rarest-first, because an image annotated with a rare class
    usually also carries common ones -- going the other way starves the tail. A
    selected image keeps *all* of its boxes, so common classes overshoot the
    budget; the realised counts are reported rather than trimmed, since dropping
    a box would teach the model that a visible animal is background.

    Args:
        image_classes: ``{image_name: {class_id, ...}}`` over the train pool.
        n_classes: Total class count.
        budget: Target images per class, or ``None`` for the whole pool.
        seed: RNG seed for the per-class shuffle.

    Returns:
        ``(selected_image_names, realised_counts_by_class_id)``.
    """
    if budget is None:
        selected = sorted(image_classes)
        counts = Counter()
        for name in selected:
            counts.update(image_classes[name])
        return selected, counts

    rng = random.Random(seed)
    pool_by_class = defaultdict(list)
    for name, classes in image_classes.items():
        for class_id in classes:
            pool_by_class[class_id].append(name)
    for class_id in pool_by_class:
        pool_by_class[class_id].sort()
        rng.shuffle(pool_by_class[class_id])

    order = sorted(range(n_classes), key=lambda c: (len(pool_by_class[c]), c))
    selected = set()
    counts = Counter()
    for class_id in order:
        for name in pool_by_class[class_id]:
            if counts[class_id] >= budget:
                break
            if name in selected:
                continue
            selected.add(name)
            counts.update(image_classes[name])

    return sorted(selected), counts


def link_or_copy(source, destination):
    """Hardlink ``source`` to ``destination``, falling back to a copy across devices."""
    if os.path.exists(destination):
        os.remove(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def build_run_dataset(root, run_dir, image_names, class_names):
    """Materialise a run's training slice and write its Ultralytics data yaml.

    val and test are pointed at the shared split directories by absolute path, so
    every budget is measured on exactly the same images.
    """
    images_dir = os.path.join(run_dir, "images", "train")
    labels_dir = os.path.join(run_dir, "labels", "train")
    for directory in (images_dir, labels_dir):
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    for image_name in image_names:
        stem = os.path.splitext(image_name)[0]
        link_or_copy(os.path.join(root, "train", "images", image_name),
                     os.path.join(images_dir, image_name))
        link_or_copy(os.path.join(root, "train", "labels", stem + ".txt"),
                     os.path.join(labels_dir, stem + ".txt"))

    yaml_path = os.path.join(run_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write("path: {}\n".format(run_dir))
        handle.write("train: {}\n".format(images_dir))
        handle.write("val: {}\n".format(os.path.join(root, "val", "images")))
        handle.write("test: {}\n".format(os.path.join(root, "test", "images")))
        handle.write("\nnames:\n")
        for class_id, name in enumerate(class_names):
            # Quoted because the BIIGLE names contain colons-adjacent punctuation
            # and non-ASCII characters.
            handle.write("  {}: {}\n".format(class_id, json.dumps(name, ensure_ascii=False)))
    return yaml_path


def epochs_for(n_images, batch, base_epochs, min_updates, max_epochs):
    """Raise the epoch count for tiny training sets so update counts stay comparable."""
    steps_per_epoch = max(1, -(-n_images // batch))
    needed = -(-min_updates // steps_per_epoch) if min_updates else 0
    return min(max(base_epochs, needed), max_epochs)


def export_nautilus_format(model, root, split, out_dir, prompt_by_class_id, conf, device,
                           chunk_size=16):
    """Re-emit test predictions in batch_inference.py's results/ + metadata.json layout.

    Boxes are already in original-image pixels, so ``metadata.json`` records the
    original size as the input size and ``evaluate_detections.py`` rescales by 1.

    The source list is fed in chunks: handed the whole split at once, Ultralytics
    sizes the batch to the source length and tries to allocate ~16 GB. Results
    are zipped back onto the chunk paths because a list source makes Ultralytics
    report every result's ``path`` as "image0.jpg"; order is preserved.
    """
    images_dir = os.path.join(root, split, "images")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    image_paths = [os.path.join(images_dir, n) for n in sorted(os.listdir(images_dir))
                   if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]

    metadata = {"prompt": "yolo", "checkpoint": str(getattr(model, "ckpt_path", "")),
                "image_dims": {}}
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start:start + chunk_size]
        predictions = model.predict(source=chunk, conf=conf, device=device,
                                    stream=True, verbose=False)
        for path, prediction in zip(chunk, predictions):
            name = os.path.basename(path)
            height, width = prediction.orig_shape

            lines = []
            for box in prediction.boxes:
                x1, y1, x2, y2 = (round(float(v)) for v in box.xyxy[0].tolist())
                class_id = int(box.cls.item())
                lines.append(json.dumps({
                    "bbox_2d": [x1, y1, x2, y2],
                    "label": prompt_by_class_id[class_id],
                    "score": round(float(box.conf.item()), 4),
                }, ensure_ascii=False))

            stem = os.path.splitext(name)[0]
            with open(os.path.join(results_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
            metadata["image_dims"][name] = {"input_height": height, "input_width": width}

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("    exported {} test predictions to {}".format(len(image_paths), results_dir))
    return results_dir


def run_one(args, root, class_names, prompt_by_class_id, image_classes, budget, seed):
    """Train and evaluate a single (budget, seed) point of the sweep."""
    from ultralytics import YOLO

    tag = "n{}_s{}".format("full" if budget is None else budget, seed)
    run_dir = os.path.join(args.project, tag)
    os.makedirs(run_dir, exist_ok=True)

    image_names, realised = select_images(image_classes, len(class_names), budget, seed)
    yaml_path = build_run_dataset(root, run_dir, image_names, class_names)
    epochs = epochs_for(len(image_names), args.batch, args.epochs,
                        args.min_updates, args.max_epochs)

    starved = [class_names[c] for c in range(len(class_names))
               if budget is not None and realised[c] < budget]
    print("\n=== {} : {} train images, {} epochs ===".format(tag, len(image_names), epochs))
    if starved:
        print("    {} class(es) below the budget (pool exhausted): {}".format(
            len(starved), ", ".join(starved[:6]) + (" ..." if len(starved) > 6 else "")))

    model = YOLO(args.weights, task="detect")
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=os.path.join(tag, "train"),
        exist_ok=True,
        seed=seed,
        patience=args.patience,
        workers=args.workers,
        pretrained=not args.from_scratch,
        val=True,
        plots=False,
    )

    best = os.path.join(args.project, tag, "train", "weights", "best.pt")
    model = YOLO(best)
    metrics = model.val(data=yaml_path, split="test", imgsz=args.imgsz,
                        batch=args.batch, device=args.device,
                        project=args.project, name=os.path.join(tag, "test"),
                        exist_ok=True, plots=False, verbose=False)

    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index.tolist()):
        per_class[class_names[class_id]] = {
            "AP50": float(metrics.box.ap50[index]),
            "AP50_95": float(metrics.box.ap[index].mean()),
        }

    export_dir = os.path.join(run_dir, "nautilus_format")
    export_nautilus_format(model, root, "test", export_dir, prompt_by_class_id,
                           args.pred_conf, args.device, args.pred_batch)

    record = {
        "budget": "full" if budget is None else budget,
        "seed": seed,
        "train_images": len(image_names),
        "epochs": epochs,
        "weights": best,
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50_95": float(metrics.box.map),
        "per_class": per_class,
        "realised_images_per_class": {class_names[c]: realised[c]
                                      for c in range(len(class_names))},
        "classes_below_budget": starved,
        "nautilus_format_dir": export_dir,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print("    test mAP50={:.4f}  mAP50-95={:.4f}".format(
        record["test_mAP50"], record["test_mAP50_95"]))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Split built by nautilus_zeroshot.py --prepare.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="Where per-run directories are written.")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS,
                        help="Comma-separated training images per class; 'full' = all.")
    parser.add_argument("--seeds", default="0", help="Comma-separated repetition seeds.")
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from random init instead of COCO-pretrained weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--min-updates", type=int, default=3000,
                        help="Raise epochs on small budgets until this many optimiser "
                             "steps are reached (0 disables).")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pred-conf", type=float, default=0.25,
                        help="Confidence threshold for the exported comparison predictions.")
    parser.add_argument("--pred-batch", type=int, default=16,
                        help="Images per predict() call when exporting predictions.")
    parser.add_argument("--summary", default=None,
                        help="Sweep summary JSON (default: <project>/sweep_results.json).")
    args = parser.parse_args()

    root = args.root
    for required in ("classes.txt", "prompt_names.csv", "train", "val", "test"):
        if not os.path.exists(os.path.join(root, required)):
            raise SystemExit(
                "{} not found under {} -- run nautilus_zeroshot.py --prepare first, "
                "then hardlink the split into this container's tree".format(required, root))

    class_names, prompt_by_class_id, image_classes = load_split(root)
    print("{} classes, {} images in the train pool".format(len(class_names), len(image_classes)))

    budgets = [None if b.strip().lower() == "full" else int(b)
               for b in args.budgets.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.project, exist_ok=True)

    summary_path = args.summary or os.path.join(args.project, "sweep_results.json")
    records = []
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            records = json.load(handle)

    for budget in budgets:
        for seed in seeds:
            record = run_one(args, root, class_names, prompt_by_class_id,
                             image_classes, budget, seed)
            label = record["budget"]
            records = [r for r in records
                       if not (r["budget"] == label and r["seed"] == seed)]
            records.append(record)
            records.sort(key=lambda r: (float("inf") if r["budget"] == "full" else r["budget"],
                                        r["seed"]))
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2, ensure_ascii=False)

    print("\n{:>8}{:>8}{:>10}{:>12}{:>14}".format(
        "budget", "seed", "imgs", "mAP@0.5", "mAP@0.5:0.95"))
    for record in records:
        print("{:>8}{:>8}{:>10}{:>12.4f}{:>14.4f}".format(
            record["budget"], record["seed"], record["train_images"],
            record["test_mAP50"], record["test_mAP50_95"]))
    print("\nsummary: {}".format(summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
