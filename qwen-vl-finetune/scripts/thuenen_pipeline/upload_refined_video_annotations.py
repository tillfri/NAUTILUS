"""Push ``thuenen_refined`` boxes onto the cloned *videos* in a BIIGLE project.

``upload_refined_to_biigle.py`` put the refined test boxes on the extracted still
frames. ``clone_test_video_volumes.py`` then cloned the 20 source videos those
frames were cut from into project 5293, annotation-free, so a reviewer can watch
the animal move instead of judging a frozen frame. This script closes the loop:
the same boxes, on the videos, as single-frame video annotations.

Three things differ from the image upload, and each one shapes the script:

* **A frame index is not a timestamp.** ``build_dataset.py`` named every frame
  ``<stem>(<video id>)_f<frame index>`` with ``frame_index = int(t * fps)`` and
  decoded at ``frame_index / fps``; fps was never written down. So the fps is read
  back off the local video with ``build_dataset.FrameReader`` -- the same class,
  the same ``average_rate``, because ``guessed_rate`` (30000/1001) differs in the
  fourth decimal and by frame 21237 that is a whole frame of drift.
* **A video annotation has no confidence.** The image upload marked the machine's
  boxes twice, by label and by a confidence below 1.0; ``POST videos/:id/annotations``
  has no such field. The label carries it alone -- which costs nothing, because
  every ``added`` box is ``Unidentified organism`` anyway, but the script asserts
  that rather than assuming it.
* **There is no bulk endpoint.** One annotation per request, ~2500 requests, so the
  run is long and the resume gate has to be exact: a request that fails leaves a
  frame half-annotated, and BIIGLE has no upsert. Each record is therefore matched
  against what the video already holds by ``(time, label, points)``, not by frame.

The volume is the state, as everywhere else in this pipeline: re-running the same
command finishes an interrupted run and posts nothing twice.

Usage:
    # once: attach both label trees to the video project. The proposals tree is
    # private, so --proposal-tree authorizes project 5293 on the existing tree
    # rather than creating a second one with a second `added` label.
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/upload_refined_video_annotations.py \
        --setup --project 5293 --proposal-tree 4578

    # build and validate the payload, write nothing to BIIGLE
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/upload_refined_video_annotations.py

    # a two-frame trial, check it in the UI, then the full run
    ... upload_refined_video_annotations.py ... --commit --limit 2
    ... upload_refined_video_annotations.py ... --commit
"""

import argparse
import json
import os
import re
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from biigle_api import BiigleApi  # noqa: E402
from build_dataset import FrameReader, find_video  # noqa: E402
from merge_annotations import read_class_names  # noqa: E402
from upload_refined_to_biigle import (  # noqa: E402
    HUMAN_ORIGINS,
    PROPOSAL_LABEL_COLOR,
    PROPOSAL_LABEL_NAME,
    PROPOSAL_TREE_NAME,
    check_against_labels,
    load_provenance,
    rectangle_points,
    resolve_labels,
    shape_id_for,
)

DEFAULT_DATASET = "/workspace/datasets/thuenen_refined"
DEFAULT_CLONE_REPORT = "/workspace/runs/biigle_clone_test_videos/clone_report.json"
DEFAULT_VIDEOS = "/workspace/datasets/biigle_videos"
DEFAULT_OUT = "/workspace/runs/biigle_upload_test_videos"
# thuenen_test_video -- the project clone_test_video_volumes.py filled.
DEFAULT_PROJECT = 5293
# fish_benthos_North Sea -- the tree the Thuenen video annotations were made in.
DEFAULT_LABEL_TREE = 1458
CONFIG_NAME = "biigle_video_upload_config.json"

# The class merge_annotations.py gives every box it invented. Uploading a video
# annotation cannot express a confidence, so if this ever stops holding, the
# `added` population becomes indistinguishable from a human's call.
ADDED_CLASS_NAME = "Unidentified organism"

VIDEO_ID_RE = re.compile(r"\((\d+)\)")
FRAME_INDEX_RE = re.compile(r"_f(\d+)$")

# A frame lasts ~33 ms, so milliseconds separate any two of them; two decimals of
# a pixel round-trip through JSON unchanged.
TIME_KEY_DIGITS = 3
POINT_KEY_DIGITS = 2


def parse_frame_name(name):
    """Split ``<stem>(<video id>)_f<frame index>.jpg`` into its two numbers."""
    stem = os.path.splitext(name)[0]
    video_ids = VIDEO_ID_RE.findall(stem)
    frame = FRAME_INDEX_RE.search(stem)
    if not video_ids or frame is None:
        raise SystemExit(
            "cannot read a video ID and frame index out of {!r} -- the frame was "
            "not named by build_dataset.py".format(name)
        )
    # The last group wins, in case a video's own stem carries parentheses.
    return int(video_ids[-1]), int(frame.group(1))


def index_frames(records_by_image):
    """Return ``{frame name: (video id, frame index)}`` for every frame."""
    return {name: parse_frame_name(name) for name in records_by_image}


def load_clone_report(path):
    """Return the clone report ``clone_test_video_volumes.py`` wrote."""
    if not os.path.exists(path):
        raise SystemExit(
            "missing {} -- run clone_test_video_volumes.py first".format(path)
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_clone_videos(api, report, video_ids, project_id):
    """Map each source video ID to the video ID of its clone in our project.

    A clone keeps the filename and gets a fresh ID, so the filename is the join:
    the report says which source video carried which name, and the clone volume's
    filename map says which new ID carries it now.
    """
    if report.get("project_id") not in (None, project_id):
        raise SystemExit(
            "clone report was written for project {}, but --project is {}".format(
                report["project_id"], project_id
            )
        )
    wanted = set(video_ids)
    clone_ids = {}
    source_volume = {}
    for volume_id, entry in sorted(report.get("volumes", {}).items()):
        names = {int(v): n for v, n in entry.get("videos", {}).items()}
        if not wanted & set(names):
            continue
        clone_id = entry.get("clone_id")
        if not clone_id:
            raise SystemExit(
                "source volume {} ({}) was never cloned -- re-run "
                "clone_test_video_volumes.py --commit".format(volume_id, entry.get("name"))
            )
        filenames = api.volume_filenames(clone_id) or {}
        by_name = {}
        for file_id, filename in filenames.items():
            if filename in by_name:
                raise SystemExit(
                    "clone volume {} has two files named {!r} -- the filename is "
                    "not a usable key".format(clone_id, filename)
                )
            by_name[filename] = int(file_id)
        for source_video, filename in sorted(names.items()):
            if source_video not in wanted:
                continue
            if filename not in by_name:
                raise SystemExit(
                    "clone volume {} (of source volume {}) holds no file named "
                    "{!r} -- the clone is incomplete".format(
                        clone_id, volume_id, filename
                    )
                )
            clone_ids[source_video] = by_name[filename]
            source_volume[source_video] = (int(volume_id), clone_id, filename)
    missing = sorted(wanted - set(clone_ids))
    if missing:
        for video_id in missing[:10]:
            print("  no clone for source video {}".format(video_id))
        raise SystemExit(
            "{} of {} videos in this split have no clone in project {}".format(
                len(missing), len(wanted), project_id
            )
        )
    return clone_ids, source_volume


def probe_videos(videos_dir, video_ids, source_volume):
    """Read fps and frame size off each source video, decoding nothing.

    ``build_dataset.py`` timed every extracted frame against ``average_rate``;
    reproducing its timestamps means reading the rate back the same way, which is
    what ``FrameReader`` does.
    """
    probes = {}
    for video_id in sorted(video_ids):
        filename = source_volume[video_id][2]
        path = find_video(videos_dir, video_id, filename)
        if path is None:
            raise SystemExit(
                "no local copy of video {} ({}) under {} -- run download_videos.py"
                .format(video_id, filename, videos_dir)
            )
        with FrameReader(path) as reader:
            probes[video_id] = {
                "path": path,
                "fps": reader.fps,
                "width": reader.width,
                "height": reader.height,
                "duration": reader.duration,
            }
    return probes


def verify_frame_sizes(dataset, split, frames, probes, frame_index):
    """Fail if a frame's JPEG is not the size of the video it was cut from.

    The boxes are normalised against the *decoded* frame, and BIIGLE stores video
    annotations in the video's pixels; the two only agree if the frame really is
    the video's own resolution. The test split is not one resolution, so this is
    checked per frame rather than assumed once.
    """
    bad = []
    for name in sorted(frames):
        video_id = frame_index[name][0]
        path = os.path.join(dataset, split, "images", name)
        with Image.open(path) as image:
            size = image.size
        probe = probes[video_id]
        if size != (probe["width"], probe["height"]):
            bad.append((name, size, (probe["width"], probe["height"])))
    if bad:
        for name, local, video in bad[:10]:
            print("  {}: frame {} vs video {}".format(name, local, video))
        raise SystemExit(
            "{} of {} frames are not the size of their source video -- the boxes "
            "cannot be scaled into video pixels".format(len(bad), len(frames))
        )
    print("verified {} frame(s) against their source video's resolution".format(
        len(frames)))


def verify_durations(api, clone_ids, probes, frame_index):
    """Fail if a frame's timestamp falls past the end of the cloned video."""
    latest = {}
    for name, (video_id, index) in frame_index.items():
        t = index / probes[video_id]["fps"]
        if t > latest.get(video_id, (-1.0, None))[0]:
            latest[video_id] = (t, name)
    bad = []
    for video_id, (t, name) in sorted(latest.items()):
        duration = api.video_info(clone_ids[video_id]).get("duration")
        if duration is not None and t > float(duration):
            bad.append((video_id, name, t, duration))
    if bad:
        for video_id, name, t, duration in bad:
            print("  video {}: {} at {:.3f}s, past the clone's {:.3f}s".format(
                video_id, name, t, duration))
        raise SystemExit(
            "{} video(s) have frames past the end of their clone -- the clone is "
            "not the video this dataset was built from".format(len(bad))
        )
    print("verified {} video(s): every timestamp is inside the clone".format(len(latest)))


def build_records(records_by_image, frame_index, clone_ids, probes, label_ids,
                  added_label_id, shape_id, class_names):
    """Turn provenance boxes into single-frame video annotation records.

    A box exists on exactly one extracted frame, so the annotation gets exactly
    one key frame: ``frames`` is ``[t]`` and ``points`` is one flat array of the
    rectangle's four vertices, nested one level as the endpoint expects.
    """
    payload = {}
    stats = {"origins": {}, "labels": {}, "skipped_degenerate": 0,
             "per_video": {}}
    misfiled = []
    for name in sorted(records_by_image):
        video_id, index = frame_index[name]
        probe = probes[video_id]
        width, height = probe["width"], probe["height"]
        timestamp = round(index / probe["fps"], 6)
        records = []
        for entry in records_by_image[name]:
            origin = entry["origin"]
            if origin in HUMAN_ORIGINS:
                label_id = label_ids[entry["fine_id"]]
            else:
                # No confidence field exists here, so the label is the only place
                # provenance can live -- and it only works while every invented
                # box shares one class that no human box of ours claims.
                if class_names[entry["fine_id"]] != ADDED_CLASS_NAME:
                    misfiled.append((name, class_names[entry["fine_id"]]))
                label_id = added_label_id
            points = rectangle_points(entry["box"], width, height)
            if points[0] >= points[2] or points[1] >= points[5]:
                stats["skipped_degenerate"] += 1
                continue
            records.append({
                "video_id": clone_ids[video_id],
                "source_video_id": video_id,
                "frame_index": index,
                "shape_id": shape_id,
                "label_id": label_id,
                "origin": origin,
                "frames": [timestamp],
                "points": [points],
            })
            stats["origins"][origin] = stats["origins"].get(origin, 0) + 1
            stats["labels"][label_id] = stats["labels"].get(label_id, 0) + 1
            key = str(video_id)
            stats["per_video"][key] = stats["per_video"].get(key, 0) + 1
        payload[name] = records
    if misfiled:
        for name, class_name in misfiled[:10]:
            print("  {}: an `added` box labelled {!r}".format(name, class_name))
        raise SystemExit(
            "{} `added` box(es) carry a class other than {!r}. A video annotation "
            "has no confidence, so the proposals label is the only thing telling a "
            "machine's guess from a human's -- and it stops working the moment an "
            "invented box claims a real class.".format(len(misfiled), ADDED_CLASS_NAME)
        )
    return payload, stats


def record_key(timestamp, label_id, points):
    """The identity of one annotation, for matching against what BIIGLE holds."""
    return (
        round(float(timestamp), TIME_KEY_DIGITS),
        int(label_id),
        tuple(round(float(p), POINT_KEY_DIGITS) for p in points),
    )


def existing_keys(api, video_ids):
    """Return ``{video id: {key: count}}`` for the annotations already posted.

    Matching on the annotation itself, rather than on "does this frame have
    anything", is what makes an interrupted run resumable: one POST creates one
    annotation, so a frame can be left with some of its boxes, and a coarser gate
    would either strand the rest or -- since BIIGLE has no upsert -- duplicate them.
    """
    counts = {}
    for video_id in sorted(video_ids):
        per_video = {}
        for annotation in api.video_annotations(video_id):
            frames = annotation.get("frames") or []
            points = annotation.get("points") or []
            if len(frames) != 1 or len(points) != 1:
                # Somebody's own multi-frame track; never ours, never a duplicate.
                continue
            for label in annotation.get("labels") or []:
                # The live API sends `label_id` beside the nested label; the
                # documented example only has the nested one.
                label_id = label.get("label_id")
                if label_id is None:
                    label_id = (label.get("label") or {}).get("id")
                if label_id is None:
                    continue
                key = record_key(frames[0], label_id, points[0])
                per_video[key] = per_video.get(key, 0) + 1
        counts[video_id] = per_video
    return counts


def pending_records(payload, existing):
    """Drop the records BIIGLE already holds, counting duplicates as multiples."""
    seen = {vid: dict(keys) for vid, keys in existing.items()}
    pending = {}
    skipped = 0
    for name in sorted(payload):
        keep = []
        for record in payload[name]:
            key = record_key(record["frames"][0], record["label_id"], record["points"][0])
            available = seen.setdefault(record["video_id"], {})
            if available.get(key):
                available[key] -= 1
                skipped += 1
                continue
            keep.append(record)
        if keep:
            pending[name] = keep
    return pending, skipped


def summarise(payload, stats, class_names, label_ids, added_label_id, probes):
    """Print what is about to be (or was) uploaded."""
    total = sum(len(r) for r in payload.values())
    frames = sum(1 for r in payload.values() if r)
    print("frames with boxes: {}   annotations: {}   videos: {}".format(
        frames, total, len(stats["per_video"])))
    for origin, count in sorted(stats["origins"].items(), key=lambda kv: -kv[1]):
        print("  origin {:<14} {}".format(origin, count))
    if stats["skipped_degenerate"]:
        print("  skipped {} degenerate box(es) with zero width or height".format(
            stats["skipped_degenerate"]))
    name_by_label = {lid: class_names[cid] for cid, lid in label_ids.items()}
    name_by_label[added_label_id] = PROPOSAL_LABEL_NAME
    print("  top labels:")
    for label_id, count in sorted(stats["labels"].items(), key=lambda kv: -kv[1])[:10]:
        print("    {:>6}  {:<48} {}".format(
            label_id, name_by_label.get(label_id, "?"), count))
    print("  fps read off each video:")
    for video_id, probe in sorted(probes.items()):
        print("    {:>6}  {:.6f} fps  {}x{}  {} box(es)".format(
            video_id, probe["fps"], probe["width"], probe["height"],
            stats["per_video"].get(str(video_id), 0)))


def do_setup(api, args):
    """Attach both label trees to the video project, idempotently.

    BIIGLE only accepts a label from a tree used by one of the *video's* projects,
    and the clones landed in a project of their own -- so the trees the image
    volume already uses have to be attached here too. Both trees exist; neither is
    created or written to.
    """
    trees = {t["id"]: t for t in api.project_label_trees(args.project)}

    if args.label_tree in trees:
        print("label tree {} ({!r}) already used by project {}".format(
            args.label_tree, trees[args.label_tree].get("name"), args.project))
    else:
        tree = api.label_tree(args.label_tree)
        api.attach_label_tree(args.project, args.label_tree)
        print("attached label tree {} ({!r}) to project {}".format(
            args.label_tree, tree.get("name"), args.project))

    proposal_tree = next(
        (t for t in api.project_label_trees(args.project)
         if t.get("name") == PROPOSAL_TREE_NAME), None
    )
    if proposal_tree is None:
        if args.proposal_tree:
            proposal_tree = api.label_tree(args.proposal_tree)
            if proposal_tree.get("name") != PROPOSAL_TREE_NAME:
                raise SystemExit(
                    "label tree {} is named {!r}, not {!r} -- refusing to attach "
                    "the wrong tree".format(args.proposal_tree,
                                            proposal_tree.get("name"),
                                            PROPOSAL_TREE_NAME))
            # The proposals tree is private, so the project has to be authorized
            # on it before it can be attached.
            api.authorize_project_for_label_tree(args.proposal_tree, args.project)
            api.attach_label_tree(args.project, args.proposal_tree)
            print("authorized and attached label tree {} ({!r}) to project {}".format(
                proposal_tree["id"], PROPOSAL_TREE_NAME, args.project))
        else:
            created = api.create_label_tree(
                PROPOSAL_TREE_NAME,
                visibility_id=2,
                project_id=args.project,
                description=("Machine-generated box proposals from "
                             "merge_annotations.py, uploaded for expert "
                             "verification. Not human-vouched."),
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

    config = {
        "project_id": args.project,
        "label_tree_id": args.label_tree,
        "proposal_tree_id": proposal_tree["id"],
        "added_label_id": added["id"],
        "shape_id": shape_id_for(api, "Rectangle"),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("wrote {}".format(path))
    return config


def load_config(args):
    """Resolve the IDs --setup wrote, falling back to the CLI."""
    path = os.path.join(args.out, CONFIG_NAME)
    config = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    if config.get("project_id") not in (None, args.project):
        raise SystemExit(
            "{} was written for project {}, but --project is {} -- use a different "
            "--out per project".format(path, config["project_id"], args.project)
        )
    added_label_id = args.added_label_id or config.get("added_label_id")
    if not added_label_id:
        raise SystemExit(
            "no label for the `added` boxes: run --setup first (it writes {}) or "
            "pass --added-label-id".format(path)
        )
    return added_label_id, config


def verify_upload(api, payload, clone_ids):
    """Re-read every touched video and check its annotations against the payload."""
    expected = {}
    for records in payload.values():
        for record in records:
            key = record_key(record["frames"][0], record["label_id"], record["points"][0])
            per_video = expected.setdefault(record["video_id"], {})
            per_video[key] = per_video.get(key, 0) + 1
    found = existing_keys(api, sorted(expected))
    missing = 0
    videos_wrong = []
    for video_id, keys in sorted(expected.items()):
        have = found.get(video_id, {})
        gap = sum(max(count - have.get(key, 0), 0) for key, count in keys.items())
        if gap:
            videos_wrong.append((video_id, gap))
            missing += gap
    total = sum(sum(v.values()) for v in found.values())
    print("{} video(s) now hold {} single-frame annotation(s)".format(
        len(found), total))
    if videos_wrong:
        for video_id, gap in videos_wrong[:10]:
            print("  video {}: {} annotation(s) of the payload not found".format(
                video_id, gap))
        raise SystemExit("{} annotation(s) are missing from BIIGLE".format(missing))
    print("all {} payload annotation(s) are present".format(
        sum(sum(v.values()) for v in expected.values())))
    source_by_clone = {c: s for s, c in clone_ids.items()}
    return {
        "annotations_in_videos": total,
        "videos_touched": len(found),
        "per_clone_video": {str(v): sum(k.values()) for v, k in sorted(found.items())},
        "clone_video_ids": {str(s): c for s, c in sorted(clone_ids.items())},
        "source_video_ids": {str(c): s for c, s in sorted(source_by_clone.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--clone-report", default=DEFAULT_CLONE_REPORT,
                        help="clone_test_video_volumes.py's report")
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS,
                        help="local copies of the source videos, read for their fps")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--project", type=int, default=DEFAULT_PROJECT)
    parser.add_argument("--label-tree", type=int, default=DEFAULT_LABEL_TREE,
                        help="tree the class labels are taken from")
    parser.add_argument("--proposal-tree", type=int, default=None,
                        help="existing proposals tree to attach in --setup")
    parser.add_argument("--added-label-id", type=int, default=None,
                        help="label for `added` boxes; default from --setup's config")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N frames, for a trial run")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="pause between requests; there are thousands of them")
    parser.add_argument("--setup", action="store_true",
                        help="attach the label trees to the video project")
    parser.add_argument("--commit", action="store_true",
                        help="actually POST; without it nothing is written to BIIGLE")
    parser.add_argument("--email", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

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
        print("--limit {}: {} frame(s)".format(args.limit, len(records_by_image)))
    check_against_labels(args.dataset, args.split, records_by_image)

    frame_index = index_frames(records_by_image)
    video_ids = sorted({v for v, _ in frame_index.values()})
    report = load_clone_report(args.clone_report)
    clone_ids, source_volume = resolve_clone_videos(api, report, video_ids, args.project)
    probes = probe_videos(args.videos_dir, video_ids, source_volume)
    verify_frame_sizes(args.dataset, args.split, records_by_image, probes, frame_index)
    verify_durations(api, clone_ids, probes, frame_index)

    class_names = read_class_names(os.path.join(args.dataset, "classes.txt"))
    label_ids = resolve_labels(api, args.label_tree, class_names)

    payload, stats = build_records(records_by_image, frame_index, clone_ids, probes,
                                   label_ids, added_label_id, shape_id, class_names)
    summarise(payload, stats, class_names, label_ids, added_label_id, probes)

    os.makedirs(args.out, exist_ok=True)
    payload_path = os.path.join(args.out, "upload_payload.json")
    with open(payload_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote {}".format(payload_path))

    if not args.commit:
        print("dry run -- nothing sent. Re-run with --commit to upload.")
        return

    existing = existing_keys(api, sorted(clone_ids.values()))
    pending, skipped = pending_records(payload, existing)
    if skipped:
        print("skipping {} annotation(s) BIIGLE already holds".format(skipped))
    if not pending:
        print("nothing left to upload")
        verify_upload(api, payload, clone_ids)
        return

    total = sum(len(r) for r in pending.values())
    print("uploading {} annotation(s) over {} frame(s), one request each".format(
        total, len(pending)))

    state_path = os.path.join(args.out, "upload_state.json")
    done_frames = []
    posted = 0
    started = time.time()
    for name in sorted(pending):
        for record in pending[name]:
            api.store_video_annotation(
                record["video_id"], record["shape_id"], record["label_id"],
                record["frames"], record["points"],
            )
            posted += 1
            if args.sleep:
                time.sleep(args.sleep)
        done_frames.append(name)
        if len(done_frames) % 25 == 0 or posted == total:
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump({"project_id": args.project, "split": args.split,
                           "frames": done_frames}, handle, indent=1)
            elapsed = time.time() - started
            rate = posted / elapsed if elapsed else 0.0
            print("  {}/{} annotation(s), {}/{} frame(s), {:.1f}/s, {:.1f} min left"
                  .format(posted, total, len(done_frames), len(pending), rate,
                          (total - posted) / rate / 60.0 if rate else 0.0))
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump({"project_id": args.project, "split": args.split,
                   "frames": done_frames}, handle, indent=1)

    result = verify_upload(api, payload, clone_ids)
    result.update({
        "project_id": args.project, "split": args.split, "dataset": args.dataset,
        "label_tree_id": args.label_tree, "added_label_id": added_label_id,
        "shape_id": shape_id, "frames_posted": len(done_frames),
        "annotations_posted": posted, "annotations_skipped": skipped,
        "origins": stats["origins"],
        "fps": {str(v): p["fps"] for v, p in sorted(probes.items())},
        "boxes_per_source_video": stats["per_video"],
    })
    report_path = os.path.join(args.out, "upload_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("wrote {}".format(report_path))


if __name__ == "__main__":
    main()
