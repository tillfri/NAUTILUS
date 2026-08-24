"""Push ``thuenen_refined`` boxes into a BIIGLE volume for expert correction.

``merge_annotations.py`` produces three kinds of box and only one of them was ever
looked at by a person. The refined geometry is a model's, and the ``added`` boxes
are a model's invention end to end. The way to turn that into a publishable dataset
is to hand it back to the annotators as *pre-drawn* boxes they correct, rather than
asking them to draw 2499 boxes again.

Two things make the upload safe to re-run and safe to read:

* **Provenance survives the round trip.** A box a human vouched for goes up at
  ``confidence`` 1.0 under its own label from the original Thünen tree; an ``added``
  box goes up at its proposal score under a separate label in a private tree, so a
  reviewer can filter the unverified population out of the volume in one click and
  never mistake a machine's guess for a colleague's call.
* **The volume itself is the state.** Before posting, the script asks BIIGLE which
  files already carry annotations and skips them. A crashed or half-finished run is
  resumed by running the same command again; there is no local state file to trust.

BIIGLE only accepts a label that belongs to a tree used by one of the image's
projects, so ``--setup`` must run once first: it attaches label tree 1458
(``fish_benthos_North Sea``, the tree the 49 BIIGLE reports came from -- all 29
names in ``classes.txt`` resolve against it exactly) and creates the private tree
holding the ``added`` label. It writes the resolved IDs to
``<out>/biigle_upload_config.json`` so the upload run never guesses them.

``POST image-annotations`` caps a request at 100 annotations. Chunks are packed on
image boundaries, so a request that fails can never leave an image half-annotated.

Needs PIL and network access. Boxes are normalised in the dataset and pixels in
BIIGLE, and the test split is *not* one resolution (1286 frames at 2704x1520, 17 at
1920x1080), so every image is measured individually; ``--verify-dims`` cross-checks
a seeded sample against the sizes BIIGLE recorded at upload time.

Usage:
    # once: attach the label tree and create the proposals label
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/upload_refined_to_biigle.py \
        --setup --project 5280 --volume 35740 \
        --out /workspace/runs/biigle_upload_test

    # build and validate the payload, write nothing to BIIGLE
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/upload_refined_to_biigle.py \
        --dataset /workspace/datasets/thuenen_refined --split test \
        --project 5280 --volume 35740 \
        --out /workspace/runs/biigle_upload_test

    # a five-image trial, then the full run
    ... upload_refined_to_biigle.py ... --commit --limit 5
    ... upload_refined_to_biigle.py ... --commit
"""

import argparse
import json
import os
import random
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from biigle_api import BULK_ANNOTATION_LIMIT, BiigleApi  # noqa: E402
from merge_annotations import read_class_names  # noqa: E402
from visualize_refined import load_labels  # noqa: E402

DEFAULT_DATASET = "/workspace/datasets/thuenen_refined"
DEFAULT_OUT = "/workspace/runs/biigle_upload_test"
DEFAULT_PROJECT = 5280
DEFAULT_VOLUME = 35740
# fish_benthos_North Sea -- the tree the Thuenen video annotations were made in.
DEFAULT_LABEL_TREE = 1458
PROPOSAL_TREE_NAME = "thuenen_refined proposals"
PROPOSAL_LABEL_NAME = "machine proposal (unverified)"
# visualize_refined.py draws `added` boxes orange; keep the colour consistent.
PROPOSAL_LABEL_COLOR = "ff9600"
CONFIG_NAME = "biigle_upload_config.json"
# Origins whose object a human marked, whatever the geometry ended up being.
HUMAN_ORIGINS = ("gt_unchanged", "gt_refined")


def load_provenance(dataset, split):
    """Return ``provenance.json``'s section for one split."""
    path = os.path.join(dataset, "provenance.json")
    if not os.path.exists(path):
        raise SystemExit(
            "missing {} -- re-run merge_annotations.py with --report".format(path)
        )
    with open(path, encoding="utf-8") as handle:
        provenance = json.load(handle)
    if split not in provenance:
        raise SystemExit(
            "provenance.json has no split {!r} (has {})".format(
                split, ", ".join(sorted(provenance))
            )
        )
    return provenance[split]


def check_against_labels(dataset, split, records_by_image):
    """Fail if provenance and the written labels disagree on any box count.

    ``visualize_refined.py`` skips such a frame; here it is fatal. Uploading a
    frame whose provenance is stale would attach the wrong labels to the wrong
    geometry, and nobody downstream would be able to tell.
    """
    mismatches = []
    for name, records in sorted(records_by_image.items()):
        stem = os.path.splitext(name)[0]
        written = load_labels(os.path.join(dataset, split, "labels", stem + ".txt"))
        if len(written) != len(records):
            mismatches.append((name, len(records), len(written)))
    if mismatches:
        for name, n_prov, n_written in mismatches[:10]:
            print("  provenance {} boxes, labels {} boxes: {}".format(
                n_prov, n_written, name))
        raise SystemExit(
            "{} frame(s) disagree between provenance.json and {}/labels -- "
            "the provenance is stale, re-run merge_annotations.py --report".format(
                len(mismatches), split
            )
        )


def resolve_image_ids(api, volume_id, image_names):
    """Map local filenames to BIIGLE image IDs via the volume's filename map."""
    filenames = api.volume_filenames(volume_id)
    by_name = {}
    for image_id, name in filenames.items():
        if name in by_name:
            raise SystemExit(
                "volume {} has two files named {!r} ({} and {}) -- the filename "
                "is not a usable key".format(volume_id, name, by_name[name], image_id)
            )
        by_name[name] = int(image_id)
    missing = [n for n in image_names if n not in by_name]
    if missing:
        for name in missing[:10]:
            print("  not in volume {}: {}".format(volume_id, name))
        raise SystemExit(
            "{} of {} local images are not in volume {} -- upload them first".format(
                len(missing), len(image_names), volume_id
            )
        )
    extra = len(by_name) - len(image_names)
    if extra:
        print("note: volume {} holds {} file(s) with no counterpart in this split"
              .format(volume_id, extra))
    return by_name


def image_sizes(dataset, split, image_names):
    """Measure every image locally -- the split is not one resolution."""
    sizes = {}
    for name in image_names:
        path = os.path.join(dataset, split, "images", name)
        with Image.open(path) as image:
            sizes[name] = image.size
    return sizes


def verify_sizes(api, image_ids, sizes, sample, seed):
    """Cross-check a seeded sample of local sizes against BIIGLE's own record."""
    names = sorted(sizes)
    if sample <= 0 or not names:
        return
    rng = random.Random(seed)
    picked = rng.sample(names, min(sample, len(names)))
    bad = []
    for name in picked:
        attrs = api.image_info(image_ids[name]).get("attrs") or {}
        remote = (attrs.get("width"), attrs.get("height"))
        if remote != sizes[name]:
            bad.append((name, sizes[name], remote))
    if bad:
        for name, local, remote in bad:
            print("  {}: local {} vs BIIGLE {}".format(name, local, remote))
        raise SystemExit(
            "{} of {} sampled images differ in size between disk and BIIGLE -- "
            "the volume does not hold the files this dataset was built from"
            .format(len(bad), len(picked))
        )
    print("verified {} image size(s) against BIIGLE, all matching".format(len(picked)))


def resolve_labels(api, tree_id, class_names):
    """Map every class in ``classes.txt`` to a label ID in the tree."""
    tree = api.label_tree(tree_id)
    by_name = {}
    for label in tree.get("labels", []):
        by_name.setdefault(label["name"], []).append(label["id"])
    ambiguous = {n: ids for n, ids in by_name.items() if len(ids) > 1}
    label_ids = {}
    unresolved = []
    for class_id, name in enumerate(class_names):
        ids = by_name.get(name)
        if not ids:
            unresolved.append(name)
        elif len(ids) > 1:
            unresolved.append("{} (ambiguous: {})".format(name, ids))
        else:
            label_ids[class_id] = ids[0]
    if unresolved:
        for name in unresolved:
            print("  no unique label in tree {}: {}".format(tree_id, name))
        raise SystemExit(
            "{} of {} classes do not resolve against label tree {!r} ({})".format(
                len(unresolved), len(class_names), tree.get("name"), tree_id
            )
        )
    if ambiguous:
        print("note: tree {} has {} duplicated label name(s), none of them ours"
              .format(tree_id, len(ambiguous)))
    return label_ids


def rectangle_points(box, width, height):
    """Normalised ``[x1, y1, x2, y2]`` -> BIIGLE rectangle vertices, in pixels.

    BIIGLE wants the four corners in order, ``[x1,y1, x2,y1, x2,y2, x1,y2]``.
    """
    x1, y1, x2, y2 = box
    x1 = min(max(x1 * width, 0.0), width)
    x2 = min(max(x2 * width, 0.0), width)
    y1 = min(max(y1 * height, 0.0), height)
    y2 = min(max(y2 * height, 0.0), height)
    return [round(v, 2) for v in (x1, y1, x2, y1, x2, y2, x1, y2)]


def build_records(records_by_image, image_ids, sizes, label_ids, added_label_id,
                  shape_id, added_fallback_confidence):
    """Turn provenance boxes into BulkStoreImageAnnotations records.

    Provenance is encoded twice on purpose: ``added`` boxes carry the proposals
    label *and* a confidence below 1.0, so either one isolates the population
    nobody has verified.
    """
    payload = {}
    stats = {"origins": {}, "labels": {}, "skipped_degenerate": 0}
    for name in sorted(records_by_image):
        width, height = sizes[name]
        image_id = image_ids[name]
        records = []
        for entry in records_by_image[name]:
            origin = entry["origin"]
            if origin in HUMAN_ORIGINS:
                label_id = label_ids[entry["fine_id"]]
                confidence = 1.0
            else:
                label_id = added_label_id
                score = entry.get("score")
                score = added_fallback_confidence if score is None else float(score)
                # Strictly below 1.0, so "confidence < 1" stays a clean filter.
                confidence = round(min(max(score, 0.01), 0.99), 4)
            points = rectangle_points(entry["box"], width, height)
            if points[0] >= points[2] or points[1] >= points[5]:
                stats["skipped_degenerate"] += 1
                continue
            records.append({
                "image_id": image_id,
                "shape_id": shape_id,
                "label_id": label_id,
                "confidence": confidence,
                "points": points,
            })
            stats["origins"][origin] = stats["origins"].get(origin, 0) + 1
            stats["labels"][label_id] = stats["labels"].get(label_id, 0) + 1
        payload[name] = records
    return payload, stats


def chunk_images(payload, image_ids, limit):
    """Pack images into requests of at most ``limit`` annotations.

    Packing on image boundaries means a failed request never leaves an image
    with some of its boxes, which is what makes the skip-annotated-files gate a
    complete resume rather than an approximate one.
    """
    chunks = []
    current = []
    current_names = []
    for name in sorted(payload):
        records = payload[name]
        if not records:
            continue
        if len(records) > limit:
            raise SystemExit(
                "{} has {} boxes, more than the {}-annotation request cap".format(
                    name, len(records), limit
                )
            )
        if len(current) + len(records) > limit:
            chunks.append((current_names, current))
            current, current_names = [], []
        current.extend(records)
        current_names.append(name)
    if current:
        chunks.append((current_names, current))
    return chunks


def summarise(payload, stats, class_names, label_ids, added_label_id):
    """Print what is about to be (or was) uploaded."""
    total = sum(len(r) for r in payload.values())
    images = sum(1 for r in payload.values() if r)
    print("images with boxes: {}   annotations: {}".format(images, total))
    for origin, count in sorted(stats["origins"].items(), key=lambda kv: -kv[1]):
        print("  origin {:<14} {}".format(origin, count))
    if stats["skipped_degenerate"]:
        print("  skipped {} degenerate box(es) with zero width or height"
              .format(stats["skipped_degenerate"]))
    name_by_label = {lid: class_names[cid] for cid, lid in label_ids.items()}
    name_by_label[added_label_id] = PROPOSAL_LABEL_NAME
    top = sorted(stats["labels"].items(), key=lambda kv: -kv[1])[:10]
    print("  top labels:")
    for label_id, count in top:
        print("    {:>6}  {:<48} {}".format(
            label_id, name_by_label.get(label_id, "?"), count))


def do_setup(api, args):
    """Attach the source label tree and create the proposals label, idempotently."""
    trees = api.project_label_trees(args.project)
    attached = {t["id"]: t for t in trees}

    if args.label_tree in attached:
        print("label tree {} ({}) already used by project {}".format(
            args.label_tree, attached[args.label_tree].get("name"), args.project))
    else:
        tree = api.label_tree(args.label_tree)
        api.attach_label_tree(args.project, args.label_tree)
        print("attached label tree {} ({!r}) to project {}".format(
            args.label_tree, tree.get("name"), args.project))
        trees = api.project_label_trees(args.project)
        attached = {t["id"]: t for t in trees}

    # The proposals label lives in a tree of our own: tree 1458 is public and in
    # use by the live Thuenen projects, and a script has no business writing to it.
    proposal_tree = next(
        (t for t in trees if t.get("name") == PROPOSAL_TREE_NAME), None
    )
    if proposal_tree is None:
        created = api.create_label_tree(
            PROPOSAL_TREE_NAME,
            visibility_id=2,
            project_id=args.project,
            description=("Machine-generated box proposals from merge_annotations.py, "
                         "uploaded for expert verification. Not human-vouched."),
        )
        proposal_tree = api.label_tree(created["id"])
        print("created private label tree {} ({!r})".format(
            proposal_tree["id"], PROPOSAL_TREE_NAME))
    else:
        proposal_tree = api.label_tree(proposal_tree["id"])
        print("reusing label tree {} ({!r})".format(
            proposal_tree["id"], PROPOSAL_TREE_NAME))

    added = next(
        (l for l in proposal_tree.get("labels", []) if l["name"] == PROPOSAL_LABEL_NAME),
        None,
    )
    if added is None:
        created = api.create_label(
            proposal_tree["id"], PROPOSAL_LABEL_NAME, PROPOSAL_LABEL_COLOR
        )
        added = created[0] if isinstance(created, list) else created
        print("created label {} ({!r})".format(added["id"], PROPOSAL_LABEL_NAME))
    else:
        print("reusing label {} ({!r})".format(added["id"], PROPOSAL_LABEL_NAME))

    shape_id = shape_id_for(api, "Rectangle")
    config = {
        "project_id": args.project,
        "volume_id": args.volume,
        "label_tree_id": args.label_tree,
        "proposal_tree_id": proposal_tree["id"],
        "added_label_id": added["id"],
        "shape_id": shape_id,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("wrote {}".format(path))
    return config


def shape_id_for(api, name):
    """Resolve a shape name to its ID rather than hardcoding the number."""
    for shape in api.shapes():
        if shape["name"] == name:
            return shape["id"]
    raise SystemExit("BIIGLE reports no shape named {!r}".format(name))


def load_config(args):
    """Resolve the IDs --setup wrote, falling back to the CLI/live API."""
    path = os.path.join(args.out, CONFIG_NAME)
    config = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    if config.get("volume_id") not in (None, args.volume):
        raise SystemExit(
            "{} was written for volume {}, but --volume is {} -- use a different "
            "--out per volume".format(path, config["volume_id"], args.volume)
        )
    added_label_id = args.added_label_id or config.get("added_label_id")
    if not added_label_id:
        raise SystemExit(
            "no label for the `added` boxes: run --setup first (it writes {}) or "
            "pass --added-label-id".format(path)
        )
    return added_label_id, config


def verify_upload(api, args, payload, image_ids):
    """Re-read the volume and check every uploaded image against the payload."""
    counts = {}
    for annotation in api.volume_annotations(args.volume):
        counts[annotation["image_id"]] = counts.get(annotation["image_id"], 0) + 1
    wrong = []
    for name, records in sorted(payload.items()):
        expected = len(records)
        actual = counts.get(image_ids[name], 0)
        if actual != expected:
            wrong.append((name, expected, actual))
    total = sum(counts.values())
    print("volume {} now holds {} annotation(s) across {} image(s)".format(
        args.volume, total, len(counts)))
    if wrong:
        for name, expected, actual in wrong[:10]:
            print("  {}: expected {}, found {}".format(name, expected, actual))
        raise SystemExit("{} image(s) do not match the payload".format(len(wrong)))
    print("all {} image(s) match the payload".format(len(payload)))
    return {"annotations_in_volume": total, "images_in_volume": len(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--project", type=int, default=DEFAULT_PROJECT)
    parser.add_argument("--volume", type=int, default=DEFAULT_VOLUME)
    parser.add_argument("--label-tree", type=int, default=DEFAULT_LABEL_TREE,
                        help="tree the class labels are taken from")
    parser.add_argument("--added-label-id", type=int, default=None,
                        help="label for `added` boxes; default from --setup's config")
    parser.add_argument("--chunk", type=int, default=BULK_ANNOTATION_LIMIT,
                        help="annotations per request (BIIGLE caps this at 100)")
    parser.add_argument("--verify-dims", type=int, default=20,
                        help="images to size-check against BIIGLE; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N images, for a trial run")
    parser.add_argument("--added-confidence", type=float, default=0.5,
                        help="confidence for an `added` box with no score")
    parser.add_argument("--setup", action="store_true",
                        help="attach the label tree and create the proposals label")
    parser.add_argument("--commit", action="store_true",
                        help="actually POST; without it nothing is written to BIIGLE")
    parser.add_argument("--email", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    if args.chunk > BULK_ANNOTATION_LIMIT:
        raise SystemExit("--chunk cannot exceed {}".format(BULK_ANNOTATION_LIMIT))

    api = BiigleApi(email=args.email, token=args.token)
    user = api.whoami()
    print("authenticated as {} {} <{}>".format(
        user.get("firstname"), user.get("lastname"), user.get("email")))

    if args.setup:
        do_setup(api, args)
        return

    added_label_id, config = load_config(args)
    shape_id = config.get("shape_id") or shape_id_for(api, "Rectangle")

    records_by_image = load_provenance(args.dataset, args.split)
    if args.limit:
        keep = sorted(records_by_image)[:args.limit]
        records_by_image = {k: records_by_image[k] for k in keep}
        print("--limit {}: {} image(s)".format(args.limit, len(records_by_image)))
    check_against_labels(args.dataset, args.split, records_by_image)

    class_names = read_class_names(os.path.join(args.dataset, "classes.txt"))
    label_ids = resolve_labels(api, args.label_tree, class_names)
    image_ids = resolve_image_ids(api, args.volume, sorted(records_by_image))
    sizes = image_sizes(args.dataset, args.split, sorted(records_by_image))
    verify_sizes(api, image_ids, sizes, args.verify_dims, args.seed)

    payload, stats = build_records(records_by_image, image_ids, sizes, label_ids,
                                   added_label_id, shape_id, args.added_confidence)
    summarise(payload, stats, class_names, label_ids, added_label_id)

    os.makedirs(args.out, exist_ok=True)
    payload_path = os.path.join(args.out, "upload_payload.json")
    with open(payload_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote {}".format(payload_path))

    if not args.commit:
        print("dry run -- nothing sent. Re-run with --commit to upload.")
        return

    annotated = set(api.volume_annotated_files(args.volume))
    pending = {n: r for n, r in payload.items()
               if r and image_ids[n] not in annotated}
    skipped = len([n for n, r in payload.items()
                   if r and image_ids[n] in annotated])
    if skipped:
        print("skipping {} image(s) that already carry annotations".format(skipped))
    if not pending:
        print("nothing left to upload")
        verify_upload(api, args, {n: r for n, r in payload.items() if r}, image_ids)
        return

    chunks = chunk_images(pending, image_ids, args.chunk)
    total = sum(len(r) for r in pending.values())
    print("uploading {} annotation(s) over {} image(s) in {} request(s)".format(
        total, len(pending), len(chunks)))

    done_images = []
    done_annotations = 0
    state_path = os.path.join(args.out, "upload_state.json")
    for index, (names, records) in enumerate(chunks, 1):
        api.bulk_store_image_annotations(records)
        done_images.extend(names)
        done_annotations += len(records)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump({"volume_id": args.volume, "split": args.split,
                       "images": done_images}, handle, indent=1)
        print("  [{}/{}] {} annotation(s), {} image(s) done".format(
            index, len(chunks), done_annotations, len(done_images)))

    report = verify_upload(api, args, {n: r for n, r in payload.items() if r}, image_ids)
    report.update({
        "volume_id": args.volume, "project_id": args.project, "split": args.split,
        "dataset": args.dataset, "label_tree_id": args.label_tree,
        "added_label_id": added_label_id, "shape_id": shape_id,
        "images_posted": len(done_images), "annotations_posted": done_annotations,
        "images_skipped": skipped, "origins": stats["origins"],
    })
    report_path = os.path.join(args.out, "upload_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote {}".format(report_path))


if __name__ == "__main__":
    main()
