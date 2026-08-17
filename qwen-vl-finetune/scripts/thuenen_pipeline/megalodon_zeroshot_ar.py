"""Frozen FathomNet Megalodon YOLOv8x, zero-shot, on the SOR-922-AR (Asterias rubens) set.

Same detector and rationale as ``megalodon_zeroshot.py`` (see that docstring for the
FathomNet/Megalodon background and the confidence-sweep-is-free trick), pointed at a
different, unrelated dataset: SOR-922-AR is a single-class "starfish" (Seestern /
Asterias rubens) set with 922 images, laid out flat as ``full_dataset/{images,labels}``
-- there is no train/val/test split to respect here, unlike the Thünen scaling split.

Because both the GT and Megalodon's own head are effectively single-class, boxes are
relabelled ``starfish`` rather than ``object`` (default in the Thünen script): the
class-aware and class-agnostic blocks of ``evaluate_detections.py`` then agree, and the
numbers read directly against the existing NAUTILUS zero-shot run in
``runs/asterias_rubens_with_class`` (prompt "Possible Objects are starfish").

Usage:
    # inside the brackish container, once (dataset + weights); weights are already
    # present from the Thünen megalodon run, no download needed
    #   cp -al /home/tfricke/nautilus/datasets/SOR-922-AR \
    #          /home/tfricke/brackish/container/SOR-922-AR
    docker exec brackish python /usr/src/ultralytics/brackish/megalodon_zeroshot_ar.py \
        --root /usr/src/ultralytics/brackish/SOR-922-AR \
        --weights /usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt \
        --device 0

To score a run, copy the exported predictions to the nautilus-qwen container (it does
not mount the brackish tree) and run the usual evaluator against the SOR-922-AR GT,
which is already reachable there via the ~/nautilus/datasets mount:

    cp -r /home/tfricke/brackish/container/sor922ar_megalodon_runs/megalodon_i1280_c025 \
          /home/tfricke/nautilus/runs/sor922ar_megalodon_i1280_c025
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/SOR-922-AR/full_dataset/labels \
      --pred-dir     /workspace/runs/sor922ar_megalodon_i1280_c025/results \
      --image-dir    /workspace/datasets/SOR-922-AR/full_dataset/images \
      --classes-file /workspace/datasets/SOR-922-AR/classes.txt \
      --save-json    /workspace/runs/sor922ar_megalodon_i1280_c025/metrics.json

With a matching single label on both sides, ``class_aware`` and ``class_agnostic`` are
the same number here -- unlike the 29-class Thünen script, there is no per-class block
worth ignoring.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from megalodon_zeroshot import conf_tag, summarise
from yolo_scaling import IMAGE_EXTENSIONS, predict_split, write_nautilus_format

DEFAULT_ROOT = "/usr/src/ultralytics/brackish/SOR-922-AR"
DEFAULT_WEIGHTS = "/usr/src/ultralytics/brackish/mbari-megalodon-yolov8x.pt"
DEFAULT_PROJECT = "/usr/src/ultralytics/brackish/sor922ar_megalodon_runs"
DEFAULT_SPLIT = "full_dataset"
# Same compressed-score grid as the Thünen script; kept identical for comparability.
DEFAULT_CONFS = "0.01,0.05,0.10,0.15,0.25,0.40"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="SOR-922-AR root (container path)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="subdirectory of --root holding images/ + labels/; "
                             "SOR-922-AR has only 'full_dataset', no train/val/test")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Megalodon's training resolution; do not lower without a reason")
    parser.add_argument("--confs", default=DEFAULT_CONFS,
                        help="comma-separated thresholds; the lowest drives the single GPU pass")
    parser.add_argument("--label", default="starfish",
                        help="constant label written for every box, matched to the "
                             "single SOR-922-AR GT class; pass an empty string to keep "
                             "the checkpoint's own names")
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
        raise SystemExit("missing {} -- hardlink SOR-922-AR into the brackish tree with "
                         "cp -al first".format(images_dir))
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
        print("label      : every box written as {!r}".format(args.label))
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
            prompt="megalodon-sor922ar", min_score=conf)
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
