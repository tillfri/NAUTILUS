"""YOLOv8x @1280 budget sweep from two initialisations: COCO vs. FathomNet Megalodon.

Follow-up #2 from ``megalodon-classagnostic.md``. That note showed a *frozen*
Megalodon finds 86% of the annotated animals (centre-in-box) on frames NAUTILUS finds
38% of, without ever having seen a Thünen image. This script asks the next question:
**does that marine representation transfer, i.e. is Megalodon worth training data?**

Two arms, identical in every respect except the weights they start from:

    coco  yolov8x.pt                     COCO, 80 classes,  68.2M params
    mega  mbari-megalodon-yolov8x.pt     FathomNet, 1 class 'object', same architecture

Both at ``imgsz 1280`` over the budgets ``1,2,4,8,16,32,64,128,full`` images/class, on
the same fixed val/test split as ``yolo_scaling.py``. The COCO arm is the control the
note asked for: it holds model size *and* resolution fixed, so any gap between the two
curves is attributable to the pretraining corpus alone. Read the gap in images/class --
"Megalodon init is worth N images per class" -- and where it closes.

Everything reusable is imported from ``yolo_scaling.py`` (budget selection, dataset
materialisation, epoch schedule, prediction, run-directory writing) and
``megalodon_zeroshot.py`` (confidence tagging and box statistics), so the training
slices and output layout are bit-for-bit the ones the published curves used.

Three things worth knowing before changing anything here:

* **A 1-class checkpoint loads into a 29-class model cleanly, but only by accident of
  naming.** ``BaseModel.load`` calls ``_remap_cls_by_names`` first; Megalodon's single
  class is named ``object`` and matches none of the 29 BIIGLE names, so no head rows are
  remapped and only the final class-logit conv (out_channels 1 vs 29) is dropped. COCO's
  80 names likewise match none, so the two arms are treated symmetrically. If a future
  label space ever contains a literal "object", the mega arm would silently inherit one
  pretrained row and the arms would stop being comparable.
* **``--imgsz`` must stay explicit.** ``Model._reset_ckpt_args`` carries
  ``{imgsz, data, task, single_cls}`` out of a checkpoint's ``train_args`` into
  ``overrides``, and Megalodon's checkpoint says ``imgsz: 640`` with a FathomNet ``data:``
  path. Explicit kwargs to ``train()``/``predict()`` win, which is the only reason this
  is safe.
* **Batch 4, not 16.** YOLOv8x at 1280 is ~13x the per-image cost of YOLOv8m at 640;
  batch 4 is about the activation footprint of the m-sweep's batch 16 at 640 and fits a
  24 GB card. Fixed rather than ``batch=-1`` autobatch, which would pick a different size
  per run and make the arms non-comparable. Note the consequence for the write-up: at the
  same ``--min-updates`` these arms see a quarter of the images the m@640 curve did.

Test predictions are exported as a **confidence sweep** rather than at one threshold:
``evaluate_detections.py`` ignores the ``score`` field and computes AP at a single
operating point, so a scored detector's mAP is a function of its threshold. The detector
runs once at the lowest threshold and the other run directories are filtered out of those
boxes in memory, so the sweep costs nothing. ``c025`` is the threshold directly
comparable with the existing ``thuenen_yolo_n*_s0`` runs.

Usage:
    # inside the brackish container, after
    #   cp -al /home/tfricke/nautilus/datasets/thuenen_scaling \
    #          /home/tfricke/brackish/container/thuenen_scaling
    #   cp .../scripts/thuenen_pipeline/{yolo_scaling,megalodon_zeroshot,\
yolo_init_comparison}.py /home/tfricke/brackish/container/
    #   curl -L -o /home/tfricke/brackish/container/yolov8x.pt \
    #     https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt
    docker exec brackish python /usr/src/ultralytics/brackish/yolo_init_comparison.py \
        --device 0

    # shard the sweep over the four GPUs (each shard writes its own --summary)
    docker exec brackish python .../yolo_init_comparison.py \
        --arms coco=/usr/src/ultralytics/brackish/yolov8x.pt \
        --budgets full,4,8 --device 0 --summary .../sweep_g0.json &
    docker exec brackish python .../yolo_init_comparison.py \
        --arms mega=/usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt \
        --budgets full,4,8 --device 1 --summary .../sweep_g1.json &
    ...

Scoring runs in the nautilus-qwen container, which does not mount the brackish tree.
The script writes ``<project>/score_commands.sh`` with every ``cp -r`` and scorer
invocation spelled out -- 18 runs x 7 thresholds is 126 of them -- so run that rather
than typing them:

    bash /home/tfricke/brackish/container/thuenen_scaling_x1280_runs/score_commands.sh
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from megalodon_zeroshot import conf_tag, summarise
from yolo_scaling import (IMAGE_EXTENSIONS, build_run_dataset, epochs_for, load_split,
                          predict_split, select_images, write_nautilus_format)

DEFAULT_ROOT = "/usr/src/ultralytics/brackish/thuenen_scaling"
DEFAULT_PROJECT = "/usr/src/ultralytics/brackish/thuenen_scaling_x1280_runs"
DEFAULT_ARMS = ("coco=/usr/src/ultralytics/brackish/yolov8x.pt,"
                "mega=/usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt")
DEFAULT_BUDGETS = "1,2,4,8,16,32,64,128,full"
# Mirrors megalodon_zeroshot.py's grid plus 0.50, the threshold at which Megalodon
# matched NAUTILUS's box budget. 0.25 is the canonical point: it is what the published
# thuenen_yolo_n*_s0 runs were exported at.
DEFAULT_EXPORT_CONFS = "0.01,0.05,0.10,0.15,0.25,0.40,0.50"
CANONICAL_CONF = 0.25

# The brackish container mounts only ~/brackish/container, so score_commands.sh has to
# translate its own paths back to the host and then into nautilus-qwen's tree.
BRACKISH_CONTAINER_PREFIX = "/usr/src/ultralytics/brackish"
BRACKISH_HOST_PREFIX = "/home/tfricke/brackish/container"
NAUTILUS_RUNS_HOST = "/home/tfricke/nautilus/runs"
NAUTILUS_RUNS_CONTAINER = "/workspace/runs"
NAUTILUS_DATASET = "/workspace/datasets/thuenen_scaling"
NAUTILUS_SCRIPTS = "/workspace/NAUTILUS/qwen-vl-finetune/scripts"
RUN_PREFIX = "thuenen_yolox1280"


def parse_arms(spec):
    """``"coco=yolov8x.pt,mega=/path/meg.pt"`` -> ``[("coco", "yolov8x.pt"), ...]``.

    Order is preserved because it drives the column order of the comparison table.
    """
    arms = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, separator, weights = chunk.partition("=")
        if not separator or not name.strip() or not weights.strip():
            raise SystemExit("--arms entries must look like name=/path/to/weights.pt, "
                             "got {!r}".format(chunk))
        arms.append((name.strip(), weights.strip()))
    if not arms:
        raise SystemExit("--arms is empty")
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate arm names: {}".format(names))
    return arms


def slice_fingerprint(image_names):
    """sha1 over the selected training filenames.

    The whole experiment rests on both arms training on the *same* images. That is
    guaranteed today -- ``select_images`` is a pure function of (pool, budget, seed) --
    but it is guaranteed silently, so record it and let ``main`` cross-check the arms.
    """
    return hashlib.sha1("\n".join(sorted(image_names)).encode("utf-8")).hexdigest()


def run_tag(arm, budget, seed):
    return "{}_n{}_s{}".format(arm, "full" if budget is None else budget, seed)


def export_conf_sweep(model, args, root, out_dir, prompt_by_class_id, weights, arm):
    """One GPU pass over the test split, materialised at every requested threshold.

    Returns ``(conf_sweep_records, boxes_at_lowest_threshold)``. Each threshold gets its
    own ``<out_dir>/<conf_tag>/`` holding ``results/`` + ``metadata.json``, which is the
    layout ``evaluate_detections.py`` and ``localization_report.py`` expect (they read
    ``image_dims`` from the pred-dir's *parent*).
    """
    confs = sorted(float(c) for c in args.export_confs.split(",") if c.strip())
    if not confs:
        raise SystemExit("--export-confs is empty")
    tags = {conf: conf_tag(conf) for conf in confs}
    if len(set(tags.values())) != len(confs):
        raise SystemExit("--export-confs collide after rounding to two decimals: {}".format(confs))

    collected = predict_split(model, root, "test", confs[0], args.device,
                              args.pred_batch, imgsz=args.imgsz)
    total = sum(len(boxes) for _, _, _, boxes in collected)
    print("    predicted {} images at conf {}: {} boxes".format(
        len(collected), confs[0], total))

    sweep = []
    for conf in confs:
        conf_dir = os.path.join(out_dir, tags[conf])
        _, written = write_nautilus_format(
            conf_dir, collected, prompt_by_class_id, weights,
            prompt="yolov8x-i{}-{}".format(args.imgsz, arm), min_score=conf)
        record = {"conf": conf, "tag": tags[conf], "dir": conf_dir,
                  "boxes_written": written}
        record.update(summarise(collected, conf))
        sweep.append(record)
        print("      {:<5} {:>6} boxes  {:>5.2f}/img  {:>5.1%} empty".format(
            tags[conf], record["boxes"], record["boxes_per_image"],
            record["empty_image_frac"]))
    return sweep, total


def run_one(args, root, class_names, prompt_by_class_id, image_classes,
            arm, weights, budget, seed):
    """Train and evaluate one (arm, budget, seed) point."""
    from ultralytics import YOLO

    tag = run_tag(arm, budget, seed)
    run_dir = os.path.join(args.project, tag)
    result_path = os.path.join(run_dir, "result.json")
    if os.path.isfile(result_path) and not args.force:
        with open(result_path, encoding="utf-8") as handle:
            print("=== {} : already done, skipping (--force to redo) ===".format(tag))
            return json.load(handle)

    os.makedirs(run_dir, exist_ok=True)
    image_names, realised = select_images(image_classes, len(class_names), budget, seed)
    yaml_path = build_run_dataset(root, run_dir, image_names, class_names)
    epochs = epochs_for(len(image_names), args.batch, args.epochs,
                        args.min_updates, args.max_epochs)

    starved = [class_names[c] for c in range(len(class_names))
               if budget is not None and realised[c] < budget]
    print("\n=== {} : {} train images, {} epochs, init {} ===".format(
        tag, len(image_names), epochs, os.path.basename(weights)))
    if starved:
        print("    {} class(es) below the budget (pool exhausted): {}".format(
            len(starved), ", ".join(starved[:6]) + (" ..." if len(starved) > 6 else "")))

    model = YOLO(weights, task="detect")
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=args.batch,
        # Explicit: Megalodon's checkpoint carries imgsz 640 into overrides.
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=os.path.join(tag, "train"),
        exist_ok=True,
        seed=seed,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        val=True,
        plots=False,
    )

    best = os.path.join(run_dir, "train", "weights", "best.pt")
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
    conf_sweep, _ = export_conf_sweep(model, args, root, export_dir,
                                      prompt_by_class_id, best, arm)

    record = {
        "arm": arm,
        "init_weights": weights,
        "budget": "full" if budget is None else budget,
        "seed": seed,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_images": len(image_names),
        "train_images_sha1": slice_fingerprint(image_names),
        "epochs": epochs,
        "weights": best,
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50_95": float(metrics.box.map),
        "per_class": per_class,
        "realised_images_per_class": {class_names[c]: realised[c]
                                      for c in range(len(class_names))},
        "classes_below_budget": starved,
        "nautilus_format_dir": export_dir,
        "canonical_conf_dir": os.path.join(export_dir, conf_tag(CANONICAL_CONF)),
        "conf_sweep": conf_sweep,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print("    test mAP50={:.4f}  mAP50-95={:.4f}".format(
        record["test_mAP50"], record["test_mAP50_95"]))
    return record


def to_host(container_path):
    """brackish container path -> host path, for commands that run outside the container."""
    if container_path.startswith(BRACKISH_CONTAINER_PREFIX):
        return BRACKISH_HOST_PREFIX + container_path[len(BRACKISH_CONTAINER_PREFIX):]
    return container_path


def write_score_commands(path, records):
    """Emit the copy-across + scoring commands for every run x threshold.

    Not run automatically: the copy targets live outside this container and the scorers
    live in another one. Generated because 126 hand-typed invocations is exactly where
    this experiment would go wrong.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by yolo_init_comparison.py -- do not edit, regenerate.",
        "# Copies each run's confidence sweep out of the brackish tree and scores it in",
        "# nautilus-qwen. evaluate_detections.py gives class-aware mAP against the prompt",
        "# label space; localization_report.py gives the class-agnostic, box-convention-",
        "# robust recalls that NAUTILUS and Megalodon are compared on.",
        "set -euo pipefail",
        "",
    ]
    for record in sorted(records, key=lambda r: (r["arm"], _budget_key(r["budget"]), r["seed"])):
        run_name = "{}_{}".format(RUN_PREFIX, run_tag(
            record["arm"],
            None if record["budget"] == "full" else record["budget"],
            record["seed"]))
        source = to_host(record["nautilus_format_dir"])
        destination = os.path.join(NAUTILUS_RUNS_HOST, run_name)
        lines += [
            "# ---- {} (mAP50 {:.4f}) ----".format(run_name, record["test_mAP50"]),
            "mkdir -p {}".format(destination),
            "cp -r {}/. {}/".format(source, destination),
        ]
        for entry in record["conf_sweep"]:
            run_dir = "{}/{}/{}".format(NAUTILUS_RUNS_CONTAINER, run_name, entry["tag"])
            for script, out in (("evaluate_detections.py", "metrics.json"),
                                ("localization_report.py", "localization.json")):
                lines.append(
                    "docker exec nautilus-qwen python {scripts}/{script} \\\n"
                    "  --gt-dir       {ds}/test/labels_prompt \\\n"
                    "  --pred-dir     {run}/results \\\n"
                    "  --image-dir    {ds}/test/images \\\n"
                    "  --classes-file {ds}/classes_prompt.txt \\\n"
                    "  --save-json    {run}/{out}".format(
                        scripts=NAUTILUS_SCRIPTS, script=script, ds=NAUTILUS_DATASET,
                        run=run_dir, out=out))
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    os.chmod(path, 0o755)
    print("wrote {}".format(path))


def _budget_key(budget):
    return float("inf") if budget == "full" else float(budget)


def check_slice_identity(records):
    """Warn if two arms at the same (budget, seed) trained on different images."""
    by_point = {}
    for record in records:
        by_point.setdefault((record["budget"], record["seed"]), {})[record["arm"]] = \
            record.get("train_images_sha1")
    for (budget, seed), fingerprints in sorted(by_point.items(), key=lambda kv: _budget_key(kv[0][0])):
        if len(set(fingerprints.values())) > 1:
            print("WARNING: budget {} seed {} trained on DIFFERENT images per arm -- the "
                  "comparison at this point is invalid: {}".format(budget, seed, fingerprints))


def print_table(records, arms):
    """budget x arm table of test mAP@0.5, with the pairwise delta when there are two arms."""
    arm_names = [name for name, _ in arms]
    grid = {}
    for record in records:
        grid.setdefault((record["budget"], record["seed"]), {})[record["arm"]] = record

    header = "{:>8}{:>6}{:>9}".format("budget", "seed", "imgs")
    for name in arm_names:
        header += "{:>14}".format(name)
    if len(arm_names) == 2:
        header += "{:>12}".format("delta")
    print("\n" + header)

    for (budget, seed) in sorted(grid, key=lambda k: (_budget_key(k[0]), k[1])):
        row = grid[(budget, seed)]
        any_record = next(iter(row.values()))
        line = "{:>8}{:>6}{:>9}".format(budget, seed, any_record["train_images"])
        for name in arm_names:
            line += "{:>14}".format("--" if name not in row
                                    else "{:.4f}".format(row[name]["test_mAP50"]))
        if len(arm_names) == 2 and all(name in row for name in arm_names):
            delta = row[arm_names[1]]["test_mAP50"] - row[arm_names[0]]["test_mAP50"]
            line += "{:>+12.4f}".format(delta)
        print(line)
    if len(arm_names) == 2:
        print("\ndelta = {} minus {}; positive means the second init helped.".format(
            arm_names[1], arm_names[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Split built by nautilus_zeroshot.py --prepare.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="Where per-run directories are written.")
    parser.add_argument("--arms", default=DEFAULT_ARMS,
                        help="Comma-separated name=weights pairs; one budget sweep each.")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS,
                        help="Comma-separated training images per class; 'full' = all.")
    parser.add_argument("--seeds", default="0", help="Comma-separated repetition seeds.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--min-updates", type=int, default=3000,
                        help="Raise epochs on small budgets until this many optimiser "
                             "steps are reached (0 disables).")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4,
                        help="4 fits YOLOv8x at 1280 on a 24 GB card; 8 will OOM.")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Megalodon's inference resolution; both arms must match.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--export-confs", default=DEFAULT_EXPORT_CONFS,
                        help="Thresholds materialised from the single prediction pass; "
                             "the lowest drives that pass.")
    parser.add_argument("--pred-batch", type=int, default=8,
                        help="Images per predict() call when exporting predictions.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run points whose result.json already exists.")
    parser.add_argument("--summary", default=None,
                        help="Sweep summary JSON (default: <project>/init_comparison.json).")
    parser.add_argument("--score-commands", default=None,
                        help="Generated scoring script (default: <project>/score_commands.sh).")
    args = parser.parse_args()

    root = args.root
    for required in ("classes.txt", "prompt_names.csv", "train", "val", "test"):
        if not os.path.exists(os.path.join(root, required)):
            raise SystemExit(
                "{} not found under {} -- run nautilus_zeroshot.py --prepare first, "
                "then hardlink the split into this container's tree".format(required, root))

    arms = parse_arms(args.arms)
    for name, weights in arms:
        if not os.path.isfile(weights) and os.path.sep in weights:
            raise SystemExit("arm {!r}: missing weights {}".format(name, weights))

    class_names, prompt_by_class_id, image_classes = load_split(root)
    n_test = len([n for n in os.listdir(os.path.join(root, "test", "images"))
                  if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS])
    print("{} classes, {} images in the train pool, {} test images".format(
        len(class_names), len(image_classes), n_test))
    print("arms: {}".format(", ".join("{} <- {}".format(n, w) for n, w in arms)))

    budgets = [None if b.strip().lower() == "full" else int(b)
               for b in args.budgets.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.project, exist_ok=True)

    summary_path = args.summary or os.path.join(args.project, "init_comparison.json")
    records = []
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            records = json.load(handle)

    for budget in budgets:
        for seed in seeds:
            for arm, weights in arms:
                record = run_one(args, root, class_names, prompt_by_class_id,
                                 image_classes, arm, weights, budget, seed)
                label = record["budget"]
                records = [r for r in records
                           if not (r["arm"] == arm and r["budget"] == label
                                   and r["seed"] == seed)]
                records.append(record)
                records.sort(key=lambda r: (_budget_key(r["budget"]), r["seed"], r["arm"]))
                with open(summary_path, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=2, ensure_ascii=False)

    check_slice_identity(records)
    print_table(records, arms)
    write_score_commands(
        args.score_commands or os.path.join(args.project, "score_commands.sh"), records)
    print("summary: {}".format(summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
