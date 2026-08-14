"""Frozen FathomNet Megalodon YOLOv8x, zero-shot and class-agnostic, on the Thünen split.

``FathomNet/megalodon-2023-yolov8`` is a YOLOv8x trained on all publicly available
FathomNet data with a **single class** ``object`` (its ``config.yaml`` is literally
``names: {0: object}``). It is a marine-domain proposal generator that has never seen
Thünen data, which makes it the localization reference the scaling experiment lacks:
it answers "how findable are these animals at all?" with the naming confound removed.

Read against ``runs/thuenen_zeroshot`` (NAUTILUS, 0 training images) and the
``runs/thuenen_yolo_n*_s0`` budget curve, all scored class-agnostically.

Two things about the metric this feeds, both of which shape the output layout:

* ``evaluate_detections.py`` **ignores the score field entirely** -- it computes
  ``AP = precision x recall`` at a single operating point. Any single mAP for a
  scored detector is therefore a function of the threshold, not of the model, so
  this script materialises a whole confidence sweep. The detector runs **once** at
  the lowest threshold and the remaining run directories are filtered out of those
  boxes in memory, so the sweep is free.
* Thünen annotations are sparse rather than exhaustive, so a class-agnostic
  detector's unmatched boxes are not evidence of false positives. Read recall
  against boxes-per-image (a proposal curve); precision is a lower bound only.

Megalodon was trained at ``imgsz 1280`` and is run there. Note that the YOLOv8m
budget curve it is compared against was trained and predicted at 640 -- that is a
resolution difference to carry into the write-up, not something to silently correct.

Usage:
    # inside the brackish container, after
    #   cp -al /home/tfricke/nautilus/datasets/thuenen_scaling \
    #          /home/tfricke/brackish/container/thuenen_scaling
    #   curl -L -o /home/tfricke/brackish/container/mbari-megalodon-yolov8x.pt \
    #     https://huggingface.co/FathomNet/megalodon-2023-yolov8/resolve/main/mbari-megalodon-yolov8x.pt
    docker exec brackish python /usr/src/ultralytics/brackish/megalodon_zeroshot.py \
        --root /usr/src/ultralytics/brackish/thuenen_scaling \
        --weights /usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt \
        --device 0

To score a run, the exported predictions have to reach the nautilus-qwen container,
which does not mount the brackish tree. They are a few MB of text:

    cp -r /home/tfricke/brackish/container/thuenen_megalodon_runs/megalodon_i1280_c025 \
          /home/tfricke/nautilus/runs/thuenen_megalodon_i1280_c025
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_megalodon_i1280_c025/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_megalodon_i1280_c025/metrics.json

Only the ``class_agnostic`` and ``localization_only_iou_0.3`` blocks of that output
are meaningful here -- ``class_aware``/``per_class`` are noise for a one-class model.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_scaling import IMAGE_EXTENSIONS, predict_split, write_nautilus_format

DEFAULT_ROOT = "/usr/src/ultralytics/brackish/thuenen_scaling"
DEFAULT_WEIGHTS = "/usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt"
DEFAULT_PROJECT = "/usr/src/ultralytics/brackish/thuenen_megalodon_runs"
# Megalodon's scores are compressed -- nothing above ~0.5 on Thünen frames -- so the
# grid is dense at the bottom and stops at 0.40 rather than the usual 0.5/0.75.
DEFAULT_CONFS = "0.01,0.05,0.10,0.15,0.25,0.40"


def conf_tag(conf):
    """0.01 -> 'c001', 0.4 -> 'c040'. Sorts lexically in threshold order."""
    return "c{:03d}".format(int(round(conf * 100)))


def summarise(collected, min_score):
    """Box statistics at one threshold -- the proposal-curve x-axis."""
    per_image = [sum(1 for _, _, s in boxes if s >= min_score)
                 for _, _, _, boxes in collected]
    n_images = len(per_image) or 1
    return {
        "boxes": sum(per_image),
        "boxes_per_image": round(sum(per_image) / n_images, 4),
        "empty_images": sum(1 for c in per_image if c == 0),
        "empty_image_frac": round(sum(1 for c in per_image if c == 0) / n_images, 4),
        "max_boxes_in_an_image": max(per_image) if per_image else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="thuenen_scaling root (container path)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Megalodon's training resolution; do not lower without a reason")
    parser.add_argument("--confs", default=DEFAULT_CONFS,
                        help="comma-separated thresholds; the lowest drives the single GPU pass")
    parser.add_argument("--label", default="object",
                        help="constant label written for every box. Forces class-agnostic "
                             "output; pass an empty string to keep the checkpoint's own names")
    parser.add_argument("--prefix", default="megalodon")
    parser.add_argument("--device", default="0")
    parser.add_argument("--pred-batch", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="re-run even if every requested run directory already exists")
    parser.add_argument("--summary", default=None,
                        help="default: <project>/megalodon_results.json")
    args = parser.parse_args()

    images_dir = os.path.join(args.root, args.split, "images")
    if not os.path.isdir(images_dir):
        raise SystemExit("missing {} -- run nautilus_zeroshot.py --prepare and hardlink the "
                         "dataset into the brackish tree with cp -al".format(images_dir))
    if not os.path.isfile(args.weights):
        raise SystemExit("missing weights {}".format(args.weights))

    confs = sorted(float(c) for c in args.confs.split(",") if c.strip())
    if not confs:
        raise SystemExit("--confs is empty")
    summary_path = args.summary or os.path.join(args.project, "megalodon_results.json")
    os.makedirs(args.project, exist_ok=True)

    tags = {conf: "{}_i{}_{}".format(args.prefix, args.imgsz, conf_tag(conf)) for conf in confs}
    if len(set(tags.values())) != len(confs):
        raise SystemExit("--confs collide after rounding to two decimals: {}".format(confs))

    done = [conf for conf in confs
            if os.path.isfile(os.path.join(args.project, tags[conf], "metadata.json"))]
    if len(done) == len(confs) and not args.force:
        print("all {} run directories already exist under {} -- nothing to do "
              "(pass --force to redo)".format(len(confs), args.project))
        return

    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    print("weights    : {}".format(args.weights))
    print("names      : {}".format(model.names))
    if args.label:
        label_by_class_id = {class_id: args.label for class_id in model.names}
        print("label      : every box written as {!r} (class-agnostic)".format(args.label))
    else:
        label_by_class_id = dict(model.names)
        print("label      : checkpoint names kept")
    if len(model.names) != 1:
        print("WARNING: checkpoint has {} classes, not 1 -- this script assumes a "
              "class-agnostic detector".format(len(model.names)))

    n_images = len([n for n in os.listdir(images_dir)
                    if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS])
    print("\n=== predicting {} {} images at imgsz {}, conf {} ===".format(
        n_images, args.split, args.imgsz, confs[0]))
    collected = predict_split(model, args.root, args.split, confs[0], args.device,
                              args.pred_batch, imgsz=args.imgsz)
    print("    collected {} boxes over {} images".format(
        sum(len(b) for _, _, _, b in collected), len(collected)))

    records = []
    for conf in confs:
        out_dir = os.path.join(args.project, tags[conf])
        results_dir, written = write_nautilus_format(
            out_dir, collected, label_by_class_id, args.weights,
            prompt="megalodon-classagnostic", min_score=conf)
        stats = summarise(collected, conf)
        record = {"tag": tags[conf], "conf": conf, "imgsz": args.imgsz,
                  "split": args.split, "weights": args.weights,
                  "results_dir": results_dir, "boxes_written": written}
        record.update(stats)
        records.append(record)
        print("  {:<28} conf {:<5} {:>6} boxes  {:>6.2f}/img  {:>5.1%} empty".format(
            tags[conf], conf, stats["boxes"], stats["boxes_per_image"],
            stats["empty_image_frac"]))

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print("\nwrote {}".format(summary_path))


if __name__ == "__main__":
    main()
