"""Find the rig artifacts Megalodon keeps re-detecting and write them as drop regions.

The Thünen sled tows two chains through the field of view, one on each side, and
Megalodon detects them as confidently as it detects animals -- up to score 0.92,
in *every* frame of a video. At ``--megalodon-add-conf 0.40`` each of those boxes
would enter ``thuenen_refined`` as a new ``unidentified organism``. This script
finds them and writes the region file ``merge_annotations.py --drop-boxes`` reads.

How an artifact is told apart from an animal: **it does not move**. The camera is
towed over the seafloor, so an animal enters the frame, crosses it and leaves --
it cannot hold the same position for hundreds of frames. A chain bolted to the
frame can, and does. So the test is per video: link proposals across frames into
clusters by IoU, and keep a cluster only if it appears in at least
``--min-persistence`` of that video's annotated frames.

Two things this deliberately does *not* do:

* **It does not draw a region around the chain and delete everything inside it.**
  The chain's bounding box covers about a third of the frame; a containment
  region there would delete every animal that swims past it. The regions written
  here carry ``"match": "iou"``, so they delete the artifact box and nothing else.
* **It does not cross video boundaries.** Every region is scoped with ``videos``
  to the video it was measured in, because the chain sits where that deployment's
  camera geometry puts it -- the left chain ends at x=0.23 in the GX videos and at
  x=0.44 in the WH489 ones.

Clustering runs on Megalodon only (NAUTILUS emits no scores, so "confident and
stationary" is not measurable there), but the regions are reported against both
runs and ``merge_annotations.py`` applies them to both.

Review the ``--render`` overlays before using the output: a region deletes a box
that no human ever looked at, and no metric will tell you it was wrong.

Torch-free except for ``--render``, which needs PIL.

Usage:
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/build_drop_regions.py \
        --dataset       /workspace/datasets/thuenen_scaling \
        --megalodon-run /workspace/runs/megalodon_proposals \
        --nautilus-run  /workspace/runs/nautilus_proposals \
        --out           /workspace/datasets/thuenen_scaling/drop_regions.json \
        --render        /workspace/runs/drop_regions_vis

    # what it would drop, without writing anything:
    python3 build_drop_regions.py ... --dry-run
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from merge_annotations import (area, containment, iou, load_gt, load_metadata,  # noqa: E402
                               load_proposals, video_of)

DEFAULT_DATASET = "/workspace/datasets/thuenen_scaling"
DEFAULT_MEGALODON = "/workspace/runs/megalodon_proposals"
DEFAULT_NAUTILUS = "/workspace/runs/nautilus_proposals"
DEFAULT_OUT = "/workspace/datasets/thuenen_scaling/drop_regions.json"
DEFAULT_SPLITS = "train,val,test"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_split(args, split):
    """``[(stem, megalodon proposals, nautilus proposals, gt boxes)]`` for one split."""
    images_dir = os.path.join(args.dataset, split, "images")
    megalodon_dims = load_metadata(args.megalodon_run, split)
    nautilus_dims = (load_metadata(args.nautilus_run, split)
                     if args.nautilus_run else {})

    frames = []
    for image_name in sorted(os.listdir(images_dir)):
        if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(image_name)[0]
        megalodon = load_proposals(os.path.join(args.megalodon_run, split), stem,
                                   megalodon_dims.get(image_name), "megalodon",
                                   args.min_score)
        nautilus = (load_proposals(os.path.join(args.nautilus_run, split), stem,
                                   nautilus_dims.get(image_name), "nautilus", 0.0)
                    if args.nautilus_run else [])
        gt_boxes = load_gt(os.path.join(args.dataset, split, "labels", stem + ".txt"),
                           os.path.join(args.dataset, split, "labels_prompt",
                                        stem + ".txt"))
        frames.append((stem, megalodon, nautilus, gt_boxes))
    return frames


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #

def cluster_video(frames, link_iou):
    """Link proposals across frames into clusters of one stationary object.

    Greedy single pass: a proposal joins the cluster it overlaps most, provided
    that overlap reaches ``link_iou``, and the cluster's representative box is the
    running mean of its members. The mean is what makes this stable -- a chain
    wobbles by a few pixels per frame and a first-member representative would
    slowly drift away from it.
    """
    clusters = []
    for stem, proposals in frames:
        for proposal in proposals:
            best, best_iou = None, link_iou
            for cluster in clusters:
                overlap = iou(proposal["box"], cluster["box"])
                if overlap >= best_iou:
                    best, best_iou = cluster, overlap
            if best is None:
                best = {"box": proposal["box"], "boxes": [], "scores": [],
                        "frames": set(), "examples": []}
                clusters.append(best)
            best["boxes"].append(proposal["box"])
            best["scores"].append(proposal["score"])
            best["frames"].add(stem)
            if len(best["examples"]) < 5:
                best["examples"].append(stem)
            count = len(best["boxes"])
            best["box"] = tuple(sum(box[i] for box in best["boxes"]) / count
                                for i in range(4))
    return clusters


def region_cost(box, frames, drop_iou, containment_threshold, add_conf):
    """What the region would delete, under exactly the rule that will delete it.

    ``frames_hit`` is recomputed here rather than taken from the cluster: greedy
    linking can leave a wobbling chain in two clusters, and what matters is not
    how the boxes were grouped but in how many frames the *final* region matches
    something. ``gt_vouched`` counts proposals a human's box would have used to
    retighten itself -- the one number that can veto a region, so it is measured
    rather than assumed.
    """
    stats = Counter()
    hit_frames = set()
    for stem, megalodon, nautilus, gt_boxes in frames:
        for source, proposals in (("megalodon", megalodon), ("nautilus", nautilus)):
            for proposal in proposals:
                if iou(proposal["box"], box) < drop_iou:
                    continue
                stats["dropped_" + source] += 1
                hit_frames.add(stem)
                score = proposal["score"]
                if score is not None and score >= add_conf:
                    stats["dropped_above_add_conf"] += 1
                if any(containment(proposal["box"], gt["box"]) >= containment_threshold
                       for gt in gt_boxes):
                    stats["gt_vouched"] += 1
    stats["frames_hit"] = len(hit_frames)
    return stats


def build_regions(args, split, frames):
    by_video = defaultdict(list)
    for entry in frames:
        by_video[video_of(entry[0])].append(entry)

    regions = []
    for video, video_frames in sorted(by_video.items()):
        frame_count = len(video_frames)
        clusters = cluster_video([(stem, megalodon)
                                  for stem, megalodon, _n, _g in video_frames],
                                 args.link_iou)

        candidates = []
        for cluster in clusters:
            if len(cluster["frames"]) < args.min_frames:
                continue
            box = cluster["box"]
            stats = region_cost(box, video_frames, args.drop_iou, args.containment,
                                args.add_conf)
            hits = stats["frames_hit"]
            if hits < args.min_frames or hits / frame_count < args.min_persistence:
                continue
            candidates.append((hits, box, cluster, stats))

        # Two clusters of the same chain converge on nearly the same mean box;
        # keeping both would double every count without dropping a single extra
        # proposal. The wider-covering one wins.
        candidates.sort(key=lambda item: -item[0])
        kept = []
        for hits, box, cluster, stats in candidates:
            if any(iou(box, other[1]) >= args.merge_iou for other in kept):
                continue
            kept.append((hits, box, cluster, stats))

        for hits, box, cluster, stats in kept:
            scores = [s for s in cluster["scores"] if s is not None]
            regions.append({
                "box": [round(v, 5) for v in box],
                "match": "iou",
                "iou": args.drop_iou,
                "videos": [video],
                "splits": [split],
                "comment": "stationary in {}/{} frames of {} (max score {:.2f})"
                           .format(hits, frame_count, video,
                                   max(scores) if scores else float("nan")),
                "stats": {
                    "frames_in_video": frame_count,
                    "frames_hit": hits,
                    "persistence": round(hits / frame_count, 3),
                    "boxes_clustered": len(cluster["boxes"]),
                    "max_score": round(max(scores), 4) if scores else None,
                    "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                    "rel_area": round(area(box), 4),
                    "dropped_megalodon": stats["dropped_megalodon"],
                    "dropped_nautilus": stats["dropped_nautilus"],
                    "dropped_above_add_conf": stats["dropped_above_add_conf"],
                    "gt_vouched": stats["gt_vouched"],
                    "examples": cluster["examples"],
                },
            })
    return regions


# --------------------------------------------------------------------------- #
# review render
# --------------------------------------------------------------------------- #

def render(args, regions_by_split, frames_by_split):
    """One frame per region, region in red, GT in green, dropped proposals in orange."""
    from PIL import Image, ImageDraw  # imported here: the rest of the script is torch- and PIL-free

    os.makedirs(args.render, exist_ok=True)
    written = 0
    for split, regions in regions_by_split.items():
        proposals_by_stem = {stem: megalodon + nautilus
                             for stem, megalodon, nautilus, _gt in frames_by_split[split]}
        gt_by_stem = {stem: gt for stem, _m, _n, gt in frames_by_split[split]}
        out_dir = os.path.join(args.render, split)
        os.makedirs(out_dir, exist_ok=True)
        for index, region in enumerate(regions):
            stem = region["stats"]["examples"][0]
            path = os.path.join(args.dataset, split, "images", stem + ".jpg")
            if not os.path.exists(path):
                continue
            image = Image.open(path).convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)

            def rectangle(box, colour, line):
                draw.rectangle([box[0] * width, box[1] * height,
                                box[2] * width, box[3] * height],
                               outline=colour, width=line)

            for gt in gt_by_stem.get(stem, []):
                rectangle(gt["box"], (0, 255, 0), 5)
            for proposal in proposals_by_stem.get(stem, []):
                if iou(proposal["box"], tuple(region["box"])) >= args.drop_iou:
                    rectangle(proposal["box"], (255, 160, 0), 5)
            rectangle(region["box"], (255, 0, 0), 8)
            draw.text((10, 10), region["comment"], fill=(255, 255, 0))

            image.thumbnail((1400, 1400))
            name = "{}_{:03d}_{}.jpg".format(split, index, region["videos"][0])
            image.save(os.path.join(out_dir, name.replace("/", "_")), quality=85)
            written += 1
    print("\nrendered {} review frames into {}".format(written, args.render))


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--megalodon-run", default=DEFAULT_MEGALODON)
    parser.add_argument("--nautilus-run", default=DEFAULT_NAUTILUS,
                        help="Only reported against, never clustered: scoreless.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--out", default=DEFAULT_OUT)

    parser.add_argument("--min-score", type=float, default=0.05,
                        help="Megalodon score a proposal needs to take part in "
                             "clustering. Low-scoring boxes on the same chain are "
                             "still dropped -- the region matches on geometry.")
    parser.add_argument("--link-iou", type=float, default=0.5,
                        help="IoU at which two boxes in different frames count as "
                             "the same standing object.")
    parser.add_argument("--min-persistence", type=float, default=0.5,
                        help="Fraction of the video's frames the region must match "
                             "a proposal in. This is the whole artifact/animal "
                             "test. Below 0.5 it buys nothing: on the test split "
                             "0.5 already removes 756 of the 758 chain boxes that "
                             "would have become annotations, and only picks up "
                             "more low-scoring ones.")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Absolute floor, so a 7-frame video cannot promote a "
                             "single lucky animal to an artifact.")
    parser.add_argument("--merge-iou", type=float, default=0.7,
                        help="Two candidate regions this close are the same "
                             "artifact; only the wider-covering one is written.")
    parser.add_argument("--add-conf", type=float, default=0.40,
                        help="merge_annotations.py's --megalodon-add-conf, used "
                             "only to report how many dropped boxes would have "
                             "become new annotations.")
    parser.add_argument("--drop-iou", type=float, default=0.5,
                        help="IoU written into the region: a proposal is dropped "
                             "when it matches the artifact box this well.")
    parser.add_argument("--containment", type=float, default=0.7,
                        help="merge_annotations.py's --containment, used only to "
                             "count proposals a human's box vouches for.")

    parser.add_argument("--render", default=None,
                        help="Directory for one review overlay per region.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report without writing the region file.")
    args = parser.parse_args()

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    regions_by_split, frames_by_split, all_regions = {}, {}, []
    for split in splits:
        frames = load_split(args, split)
        regions = build_regions(args, split, frames)
        frames_by_split[split] = frames
        regions_by_split[split] = regions
        all_regions.extend(regions)

        megalodon_total = sum(len(m) for _s, m, _n, _g in frames)
        nautilus_total = sum(len(n) for _s, _m, n, _g in frames)
        dropped_m = sum(r["stats"]["dropped_megalodon"] for r in regions)
        dropped_n = sum(r["stats"]["dropped_nautilus"] for r in regions)
        vouched = sum(r["stats"]["gt_vouched"] for r in regions)
        videos = len({r["videos"][0] for r in regions})
        print("{:5s} {:4d} images, {:3d} regions across {:2d} videos"
              .format(split, len(frames), len(regions), videos))
        print("      megalodon: {} of {} proposals dropped (>= score {})"
              .format(dropped_m, megalodon_total, args.min_score))
        print("      nautilus:  {} of {} proposals dropped"
              .format(dropped_n, nautilus_total))
        print("      {} of the dropped megalodon boxes score >= {} and would "
              "have become annotations"
              .format(sum(r["stats"]["dropped_above_add_conf"] for r in regions),
                      args.add_conf))
        print("      {} dropped proposals sit >= {} inside a GT box"
              .format(vouched, args.containment)
              + ("  <-- inspect these before writing" if vouched else ""))

    print("\n{} regions total".format(len(all_regions)))
    if all_regions:
        by_area = sorted(all_regions, key=lambda r: -r["stats"]["rel_area"])
        print("largest:  {:.3f} of the frame  {}".format(
            by_area[0]["stats"]["rel_area"], by_area[0]["comment"]))
        print("smallest: {:.3f} of the frame  {}".format(
            by_area[-1]["stats"]["rel_area"], by_area[-1]["comment"]))

    if args.render:
        render(args, regions_by_split, frames_by_split)

    if args.dry_run:
        print("\ndry run: {} not written".format(args.out))
        return 0

    payload = {
        "meta": {
            "script": os.path.basename(__file__),
            "dataset": args.dataset,
            "megalodon_run": args.megalodon_run,
            "nautilus_run": args.nautilus_run,
            "min_score": args.min_score,
            "link_iou": args.link_iou,
            "min_persistence": args.min_persistence,
            "min_frames": args.min_frames,
            "drop_iou": args.drop_iou,
            "merge_iou": args.merge_iou,
        },
        "regions": all_regions,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("\nwrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
