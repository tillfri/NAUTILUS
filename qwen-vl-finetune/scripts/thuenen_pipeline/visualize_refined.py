"""Draw a random, seeded sample of ``thuenen_refined`` frames for human review.

``merge_annotations.py`` produces three kinds of box and they need to be told apart
by eye, because only one of them was ever looked at by a person:

* **gt_refined / strict** (green) -- a human's instance and label, a model's
  geometry, matched by containment. The human's *original* box is drawn alongside
  in grey so the tightening is visible; that pair is the whole point of the
  refinement and the thing to sanity-check.
* **gt_refined / fallback** (violet) -- the same, but matched by the relaxed IoU
  pass, so the new box typically *grew* rather than shrank (median 1.2x). These are
  the ones where the human's box was drawn offset; drawn apart because a growing
  box deserves a second look in a script whose job is shrinking them.
* **gt_unchanged** (blue) -- a human's box that no proposal sat inside. Untouched.
* **added** (orange) -- invented by Megalodon above ``--megalodon-add-conf``, label
  ``unidentified organism``, **never seen by a human**. These are the boxes that
  can silently poison the dataset, so they carry their score in the caption.

Geometry comes from the written ``labels/`` files and provenance from
``provenance.json`` (written by ``merge_annotations.py --report``), keyed by image
filename. The two are cross-checked per frame: a mismatch in box count means the
provenance is stale relative to the labels and the script refuses the frame rather
than drawing a misleading overlay.

The sample is drawn with an explicit ``--seed`` so the same frames come back on a
re-run and a reviewer can refer to one by name. By default only frames carrying at
least one box are eligible -- an empty frame tells a reviewer nothing.

Needs PIL, so it runs in the container rather than on the host.

Usage:
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/visualize_refined.py \
        --dataset /workspace/datasets/thuenen_refined \
        --out     /workspace/runs/thuenen_refined_vis \
        --splits  train,val,test --limit 30 --seed 0

    # only the invented boxes, the population nobody has verified:
    python3 visualize_refined.py ... --only-origin added --limit 40
"""

import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from merge_annotations import read_class_names  # noqa: E402

DEFAULT_DATASET = "/workspace/datasets/thuenen_refined"
DEFAULT_OUT = "/workspace/runs/thuenen_refined_vis"
DEFAULT_SPLITS = "train,val,test"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# origin -> (colour, width)
STYLE = {
    "gt_refined": ((60, 220, 60), 5),
    "gt_refined_fallback": ((205, 120, 255), 5),
    "gt_unchanged": ((70, 150, 255), 5),
    "added": ((255, 150, 0), 5),
}
ORIGINAL_COLOUR = (170, 170, 170)


def load_labels(path):
    """YOLO label file -> ``[(class_id, (x1, y1, x2, y2))]`` normalised."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, width, height = (float(v) for v in parts[1:])
            out.append((cid, (cx - width / 2, cy - height / 2,
                              cx + width / 2, cy + height / 2)))
    return out


def font_for(image):
    """A font that stays readable on a 2704px frame and on a small one."""
    size = max(14, int(image.height / 55))
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def caption(draw, xy, text, colour, font):
    """Text on a filled plate, so it survives a bright seafloor."""
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except AttributeError:                       # very old PIL
        width, height = draw.textsize(text, font=font)
        left, top, right, bottom = x, y, x + width, y + height
    draw.rectangle([left - 3, top - 2, right + 3, bottom + 2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour, font=font)


def draw_frame(image_path, boxes, names, show_original):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = font_for(image)

    # the human's original box first, so a refined box is drawn on top of it
    if show_original:
        for record in boxes:
            original = record.get("gt_box")
            if not original or record["origin"] != "gt_refined":
                continue
            x1, y1, x2, y2 = original
            draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height],
                           outline=ORIGINAL_COLOUR, width=3)

    for record in boxes:
        colour, line = STYLE.get(style_key(record), ((255, 255, 255), 4))
        x1, y1, x2, y2 = record["box"]
        draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height],
                       outline=colour, width=line)
        label = names[record["fine_id"]] if record["fine_id"] < len(names) \
            else str(record["fine_id"])
        if record["origin"] == "added":
            label = "+ {} {:.2f}".format(label, record.get("score") or 0.0)
        elif record["origin"] == "gt_refined" and record.get("gt_box"):
            label = "{} {:.2f}x".format(label, _ratio(record["box"],
                                                      record["gt_box"]))
        caption(draw, (x1 * width + 4, max(0, y1 * height - font.size - 6)),
                label, colour, font)
    return image


def style_key(record):
    """``gt_refined`` splits by which pass matched it; everything else is its origin."""
    if record["origin"] == "gt_refined" and record.get("refine") == "fallback":
        return "gt_refined_fallback"
    return record["origin"]


def _ratio(box, original):
    small = (box[2] - box[0]) * (box[3] - box[1])
    big = (original[2] - original[0]) * (original[3] - original[1])
    return small / big if big > 0 else 0.0


def legend(image, counts):
    """A key in the top-left, because the colours mean nothing on their own."""
    draw = ImageDraw.Draw(image)
    font = font_for(image)
    rows = [("gt_refined", "human instance, model box by containment"),
            ("gt_refined_fallback", "same, by the relaxed IoU pass (box usually grew)"),
            ("gt_unchanged", "human box, no proposal matched it"),
            ("added", "invented by Megalodon, unverified")]
    pad, line_h = 10, font.size + 8
    box_h = pad * 2 + line_h * len(rows)
    draw.rectangle([0, 0, image.width, box_h], fill=(0, 0, 0))
    for index, (origin, description) in enumerate(rows):
        y = pad + index * line_h
        colour = STYLE[origin][0]
        draw.rectangle([pad, y + 3, pad + font.size, y + font.size], fill=colour)
        draw.text((pad + font.size + 10, y),
                  "{}  ({})  {}".format(origin, counts.get(origin, 0), description),
                  fill=colour, font=font)
    return image


def collect(dataset, provenance, splits, only_origin, require_boxes):
    """Eligible ``(split, image_name)`` pairs across the requested splits."""
    frames = []
    for split in splits:
        images_dir = os.path.join(dataset, split, "images")
        if not os.path.isdir(images_dir):
            raise SystemExit("no such split: " + images_dir)
        per_split = provenance.get(split, {})
        for name in sorted(os.listdir(images_dir)):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            records = per_split.get(name, [])
            if only_origin:
                records = [r for r in records if style_key(r) == only_origin]
            if require_boxes and not records:
                continue
            frames.append((split, name))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--provenance", default=None,
                        help="Defaults to <dataset>/provenance.json.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--limit", type=int, default=30,
                        help="How many frames to draw, sampled without replacement.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pins the sample so a re-run returns the same frames.")
    parser.add_argument("--only-origin", default=None,
                        choices=sorted(STYLE),
                        help="Restrict both the sampling and the drawing to one "
                             "kind of box, e.g. the unverified added ones.")
    parser.add_argument("--all-frames", action="store_true",
                        help="Also sample frames that carry no box at all.")
    parser.add_argument("--no-original", action="store_true",
                        help="Omit the grey pre-refinement box.")
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args()

    path = args.provenance or os.path.join(args.dataset, "provenance.json")
    if not os.path.exists(path):
        raise SystemExit("missing {} -- re-run merge_annotations.py with --report"
                         .format(path))
    with open(path, encoding="utf-8") as handle:
        provenance = json.load(handle)
    names = read_class_names(os.path.join(args.dataset, "classes.txt"))

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    frames = collect(args.dataset, provenance, splits, args.only_origin,
                     not args.all_frames)
    if not frames:
        raise SystemExit("no eligible frames")
    rng = random.Random(args.seed)
    sample = rng.sample(frames, min(args.limit, len(frames)))
    sample.sort()
    print("{} eligible frames, drawing {} (seed {})"
          .format(len(frames), len(sample), args.seed))

    os.makedirs(args.out, exist_ok=True)
    totals = {}
    for split, name in sample:
        stem = os.path.splitext(name)[0]
        records = provenance.get(split, {}).get(name, [])
        written = load_labels(os.path.join(args.dataset, split, "labels",
                                           stem + ".txt"))
        if len(written) != len(records):
            print("  [skip] {}/{}: provenance has {} boxes, labels have {}"
                  .format(split, stem, len(records), len(written)))
            continue
        if args.only_origin:
            records = [r for r in records if style_key(r) == args.only_origin]

        counts = {}
        for record in records:
            key = style_key(record)
            counts[key] = counts.get(key, 0) + 1
            totals[key] = totals.get(key, 0) + 1
        image = draw_frame(os.path.join(args.dataset, split, "images", name),
                           records, names, not args.no_original)
        if not args.no_legend:
            image = legend(image, counts)
        image.save(os.path.join(args.out, "{}_{}.jpg".format(split, stem)),
                   quality=88)

    print("boxes drawn: " + ", ".join("{} {}".format(k, v)
                                      for k, v in sorted(totals.items())))
    print("-> " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
