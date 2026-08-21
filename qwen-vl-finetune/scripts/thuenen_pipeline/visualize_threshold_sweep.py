"""Render the same images at several Megalodon thresholds, side by side on disk.

``proposal_threshold_report.py`` prices a threshold in numbers; this renders what
the numbers mean. For each threshold it materialises a filtered proposal run over a
**pinned** subsample -- the same images at every threshold, so the directories are
directly comparable -- and hands it to ``visualize_errors.py``.

Two things make the figures readable as a *merge* preview rather than as a detector
score sheet:

* Matching is **class-agnostic** (Megalodon only knows ``object``), so the labels are
  free for something more useful: every proposal is labelled by what
  ``merge_annotations.py`` would do with it at this threshold -- ``refine`` if it
  sits ``>= --containment`` inside a GT box (that box adopts its geometry), ``add``
  if it does not (it becomes a new ``unidentified organism``). Reading the panels:
  the **False Positives** tile is the cost of the threshold, the **True Positives**
  and **False Negatives** tiles are its benefit.
* ``visualize_errors.py`` still matches at IoU >= ``--iou-threshold``, which is a
  *stricter* test than containment on purpose: a tight proposal inside an inflated
  circle-tool GT box scores IoU ~0.48 and therefore shows up as an FP *and* an FN at
  the same position. Those doubled boxes are exactly the annotation defect this
  whole experiment exists to fix -- seeing them is the point.

The subsample is written once to ``<out>/subsample.json`` and then left alone, same
discipline as ``screening_subsample.json`` and ``prompt_names.csv``.

Output:
    <out>/subsample.json
    <out>/gt/<split>/                       pinned GT labels (symlinks)
    <out>/conf_0.25/<split>/results/        proposals filtered at that threshold
    <out>/conf_0.25/<split>/metadata.json
    <out>/conf_0.25/<split>/figures/*.png   the 2x2 panels
    <out>/conf_0.25/<split>/figures/errors.json

Needs matplotlib, so unlike the rest of the pipeline this one wants the container.

Usage:
    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \
       python3 visualize_threshold_sweep.py \
         --megalodon-run /workspace/runs/megalodon_proposals \
         --nautilus-run  /workspace/runs/nautilus_proposals \
         --dataset       /workspace/datasets/thuenen_scaling \
         --out           /workspace/runs/megalodon_threshold_vis \
         --thresholds 0.01,0.05,0.10,0.25,0.40 --per-split 100 --jobs 6"

    # re-render one threshold only (the subsample stays pinned):
    python3 visualize_threshold_sweep.py --thresholds 0.60 --out ... 
"""

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from merge_annotations import (  # noqa: E402
    containment, fuse_sources, load_gt, load_metadata, load_proposals,
)

SCRIPTS = os.path.dirname(HERE)
DEFAULT_DATASET = "/workspace/datasets/thuenen_scaling"
DEFAULT_MEGALODON = "/workspace/runs/megalodon_proposals"
DEFAULT_NAUTILUS = "/workspace/runs/nautilus_proposals"
DEFAULT_OUT = "/workspace/runs/megalodon_threshold_vis"
DEFAULT_THRESHOLDS = "0.01,0.05,0.10,0.25,0.40"


def pin_subsample(args, splits):
    """Choose (once) the images every threshold is rendered on."""
    path = os.path.join(args.out, "subsample.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            pinned = json.load(handle)
        print("subsample: reusing {} ({})".format(
            path, ", ".join("{} {}".format(k, len(v)) for k, v in pinned.items())))
        return pinned

    pinned = {}
    for split in splits:
        images_dir = os.path.join(args.dataset, split, "images")
        names = sorted(name for name in os.listdir(images_dir)
                       if os.path.splitext(name)[1].lower() in
                       (".jpg", ".jpeg", ".png"))
        rng = random.Random("{}:{}".format(args.seed, split))
        pinned[split] = sorted(rng.sample(names, min(args.per_split, len(names))))

    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(pinned, handle, indent=2)
    print("subsample: wrote {} ({})".format(
        path, ", ".join("{} {}".format(k, len(v)) for k, v in pinned.items())))
    return pinned


def pin_gt(args, split, names):
    """A GT directory holding only the pinned stems.

    ``evaluate_detections.load_dataset`` intersects GT and prediction stems and
    warns about every non-shared one; pointing it at the full split would bury the
    run in 1200 warnings per threshold.
    """
    gt_dir = os.path.join(args.out, "gt", split)
    os.makedirs(gt_dir, exist_ok=True)
    source_dir = os.path.join(args.dataset, split, args.label_space)
    for name in names:
        stem = os.path.splitext(name)[0] + ".txt"
        link = os.path.join(gt_dir, stem)
        if not os.path.exists(link):
            os.symlink(os.path.join(source_dir, stem), link)
    return gt_dir


def write_filtered_run(args, split, names, threshold, run_dir):
    """One proposal run at one threshold, labelled by the merge's verdict."""
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    megalodon_dims = load_metadata(args.megalodon_run, split)
    megalodon_split = os.path.join(args.megalodon_run, split)
    nautilus_dims, nautilus_split = {}, None
    if args.nautilus_run:
        nautilus_dims = load_metadata(args.nautilus_run, split)
        nautilus_split = os.path.join(args.nautilus_run, split)

    labels_dir = os.path.join(args.dataset, split, "labels")
    prompt_dir = os.path.join(args.dataset, split, "labels_prompt")

    metadata = {"prompt": "megalodon-proposals@{}".format(threshold),
                "checkpoint": args.megalodon_run, "image_dims": {}}
    counts = {"refine": 0, "add": 0}

    for name in names:
        stem = os.path.splitext(name)[0]
        dims = megalodon_dims.get(name)
        width, height = float(dims["input_width"]), float(dims["input_height"])

        megalodon = load_proposals(megalodon_split, stem, dims, "megalodon", threshold)
        nautilus = []
        if nautilus_split is not None:
            nautilus = load_proposals(nautilus_split, stem, nautilus_dims.get(name),
                                      "nautilus", 0.0)
        pool, _fused = fuse_sources(megalodon, nautilus, "megalodon", "mean",
                                    args.fuse_iou)
        gt_boxes = load_gt(os.path.join(labels_dir, stem + ".txt"),
                           os.path.join(prompt_dir, stem + ".txt"))

        lines = []
        for proposal in pool:
            if args.unmatched == "megalodon" and "megalodon" not in proposal["sources"]:
                continue
            inside = any(containment(proposal["box"], gt["box"]) >= args.containment
                         for gt in gt_boxes)
            verdict = "refine" if inside else "add"
            counts[verdict] += 1
            x1, y1, x2, y2 = proposal["box"]
            lines.append(json.dumps({
                "bbox_2d": [round(x1 * width), round(y1 * height),
                            round(x2 * width), round(y2 * height)],
                "label": verdict,
                "score": proposal["score"],
            }))

        with open(os.path.join(results_dir, stem + ".txt"), "w",
                  encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        metadata["image_dims"][name] = {"input_height": int(height),
                                        "input_width": int(width)}

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return results_dir, counts


def render(args, split, threshold, gt_dir, results_dir, figures_dir):
    command = [
        sys.executable, os.path.join(SCRIPTS, "visualize_errors.py"),
        "--gt-dir", gt_dir,
        "--pred-dir", results_dir,
        "--image-dir", os.path.join(args.dataset, split, "images"),
        "--classes-file", os.path.join(args.dataset, args.classes_file),
        "--save-dir", figures_dir,
        "--iou-threshold", str(args.iou_threshold),
        "--class-agnostic",
    ]
    result = subprocess.run(command, cwd=SCRIPTS, capture_output=True, text=True)
    if result.returncode != 0:
        print("[fail] {} conf {}\n{}".format(split, threshold, result.stderr[-2000:]))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--megalodon-run", default=DEFAULT_MEGALODON)
    parser.add_argument("--nautilus-run", default=DEFAULT_NAUTILUS,
                        help="fused into the pool as --fuse megalodon would; "
                             "pass '' to render Megalodon alone")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--per-split", type=int, default=100)
    parser.add_argument("--seed", default="thuenen-threshold-vis")
    parser.add_argument("--containment", type=float, default=0.7)
    parser.add_argument("--fuse-iou", type=float, default=0.5)
    parser.add_argument("--unmatched", default="megalodon",
                        choices=("megalodon", "both"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--label-space", default="labels_prompt",
                        choices=("labels_prompt", "labels"))
    parser.add_argument("--classes-file", default="classes_prompt.txt")
    parser.add_argument("--jobs", type=int, default=4,
                        help="(split, threshold) pairs rendered in parallel")
    args = parser.parse_args()

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    thresholds = sorted(float(v) for v in args.thresholds.split(",") if v.strip())
    pinned = pin_subsample(args, splits)

    tasks = []
    for threshold in thresholds:
        label = "conf_{:.2f}".format(threshold)
        for split in splits:
            names = pinned[split]
            run_dir = os.path.join(args.out, label, split)
            gt_dir = pin_gt(args, split, names)
            results_dir, counts = write_filtered_run(args, split, names, threshold,
                                                     run_dir)
            print("{} {}: {} images, {} refine + {} add proposals".format(
                label, split, len(names), counts["refine"], counts["add"]))
            tasks.append((split, threshold, gt_dir, results_dir,
                          os.path.join(run_dir, "figures")))

    print("\nrendering {} (split, threshold) pairs on {} workers".format(
        len(tasks), args.jobs))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = sum(pool.map(lambda task: render(args, *task), tasks))
    print("done, {} failed".format(failures))
    print("figures under {}/conf_*/<split>/figures".format(args.out))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
