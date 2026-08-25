"""Clone the source video volumes of a ``thuenen_refined`` split into our own project.

``upload_refined_to_biigle.py`` put the refined boxes in front of the experts as still
frames. A still frame is a poor thing to adjudicate on: whether a blob is an animal or a
piece of the towed sled is often only decidable by watching it move. So the reviewers need
the *videos* the frames were cut from, in the same project.

Those videos sit in three other people's projects (1745, 1761, 3764) and already carry the
original BIIGLE annotations -- which are the very labels under review. ``POST
volumes/:id/clone-to/:project_id`` is the endpoint for exactly this: ``clone_annotations``
defaults to false, so the clone lands annotation-free, and ``only_files`` narrows it to
chosen files.

Two things shape the script:

* **The filename is the link back to the video.** ``build_dataset.py`` writes frames as
  ``<video filename stem>(<video id>)_f<frame>.jpg``, so the parenthesised group in
  ``WH489_St57_DD109_2_converted(38332)_f002960.jpg`` is BIIGLE video 38332. The 1303 test
  frames come from 20 videos, which ``GET videos/:id`` resolves to 19 volumes -- 38329 and
  38330 share volume 24445. Those 19 volumes hold 35 files in total, so the clone is
  narrowed with ``only_files`` (pass ``--all-files`` to mirror the source volumes whole).
* **The target project is the state.** ``clone-to`` has no upsert -- posting the same clone
  twice makes two volumes. Before cloning, the script asks the project what it already
  holds and skips a source volume whose clone is already there, so re-running the same
  command is a complete resume. Same discipline as the upload script's annotated-files gate.

The source volumes all live on *another user's* storage disk (``user-2227://`` for 17 of
them, ``user-2092://`` for the two MGF_SAR_Hol0[12] ones). The clone copies that URL while
making us the creator, and no API call can tell you whether the disk still resolves for the
clone -- open one in the BIIGLE UI and check that the video actually streams.

Usage:
    # dry run: parse the filenames, resolve the volumes, print the plan
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/clone_test_video_volumes.py \
        --dataset /workspace/datasets/thuenen_refined --split test --project 5293 \
        --out /workspace/runs/biigle_clone_test_videos

    # clone for real; run it again afterwards and it should clone 0 and skip 19
    ... clone_test_video_volumes.py ... --commit
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from biigle_api import BiigleApi  # noqa: E402

DEFAULT_DATASET = "/workspace/datasets/thuenen_refined"
DEFAULT_OUT = "/workspace/runs/biigle_clone_test_videos"
# thuenen_test_video -- the project the cloned, annotation-free videos go into.
DEFAULT_PROJECT = 5293
# build_dataset.py names a frame `<stem>(<video id>)_f<frame>.jpg`.
VIDEO_ID_RE = re.compile(r"\((\d+)\)")


def scan_video_ids(dataset, split):
    """Return ``{video_id: frame_count}`` for one split's images.

    The video ID is the parenthesised group build_dataset.py put in the frame
    name. A stem could contain its own brackets, so the *last* match wins.

    Raises:
        SystemExit: If the directory is missing or any frame carries no ID.
    """
    images_dir = os.path.join(dataset, split, "images")
    if not os.path.isdir(images_dir):
        raise SystemExit("missing {}".format(images_dir))
    counts = {}
    unmatched = []
    total = 0
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        total += 1
        matches = VIDEO_ID_RE.findall(name)
        if not matches:
            unmatched.append(name)
            continue
        video_id = int(matches[-1])
        counts[video_id] = counts.get(video_id, 0) + 1
    if unmatched:
        raise SystemExit(
            "{} frame(s) carry no (<video id>) group, e.g.:\n  {}".format(
                len(unmatched), "\n  ".join(unmatched[:10])
            )
        )
    if not counts:
        raise SystemExit("no images found in {}".format(images_dir))
    print("{}: {} frame(s) from {} video(s)".format(split, total, len(counts)))
    return counts


def resolve_volumes(api, video_counts):
    """Group the videos by their volume.

    Returns:
        ``{volume_id: {"name", "projects", "videos", "frames", "all_files",
        "dropped"}}``, where ``videos`` maps our video IDs to filenames and
        ``dropped`` lists the volume's other files.
    """
    volumes = {}
    for video_id in sorted(video_counts):
        video = api.video_info(video_id)
        volume_id = video["volume_id"]
        entry = volumes.get(volume_id)
        if entry is None:
            info = api.volume_info(volume_id)
            entry = volumes[volume_id] = {
                "name": info["name"],
                "url": info.get("url"),
                "projects": [p["id"] for p in info.get("projects", [])],
                "videos": {},
                "frames": 0,
                "all_files": sorted(api.volume_files(volume_id)),
            }
        entry["videos"][video_id] = video.get("filename")
        entry["frames"] += video_counts[video_id]
    for entry in volumes.values():
        entry["dropped"] = [f for f in entry["all_files"] if f not in entry["videos"]]
    return volumes


def filenames_map(api, volume_id):
    """Return a volume's ``{file_id: filename}`` map, always as a dict.

    ``GET volumes/:id/filenames`` serialises an empty map as ``[]`` rather than
    ``{}`` -- PHP's json_encode cannot tell an empty associative array from an
    empty list -- and a freshly cloned volume is empty for a moment.
    """
    payload = api.volume_filenames(volume_id)
    return payload if isinstance(payload, dict) else {}


def wait_for_files(api, volume_id, expected, timeout=120.0, interval=2.0):
    """Poll a fresh clone until its files exist.

    ``clone-to`` returns the volume record before the files are copied across:
    the volume answers with an empty file list, and ``creating_async`` goes back
    to false once the job has run. Verifying without waiting reads the gap.

    Returns:
        The ``{file_id: filename}`` map, empty if the wait timed out.
    """
    deadline = time.time() + timeout
    while True:
        filenames = filenames_map(api, volume_id)
        if len(filenames) >= expected:
            return filenames
        if time.time() >= deadline:
            return filenames
        time.sleep(interval)


def existing_clones(api, project_id, volumes):
    """Return ``{source_volume_id: clone}`` for clones already in the project.

    A clone keeps the source's name, and the 19 source names are distinct -- but
    a name alone is a weak key, so the filename set has to match too. A volume
    that carries the name and not the files is a half-made clone, and saying so
    is more useful than silently cloning a second one.

    Raises:
        SystemExit: If a same-named volume holds a different set of files.
    """
    by_name = {}
    for volume in api.project_volumes(project_id):
        by_name.setdefault(volume["name"], []).append(volume)
    found = {}
    for volume_id, entry in sorted(volumes.items()):
        candidates = by_name.get(entry["name"], [])
        if not candidates:
            continue
        if len(candidates) > 1:
            raise SystemExit(
                "project {} already holds {} volumes named {!r} -- a duplicated "
                "clone; delete the extras before re-running".format(
                    project_id, len(candidates), entry["name"]
                )
            )
        clone = candidates[0]
        want = set(entry["videos"].values())
        have = set(filenames_map(api, clone["id"]).values())
        if have != want:
            raise SystemExit(
                "volume {} ({!r}) in project {} holds {} instead of the expected "
                "{} -- resolve it by hand before re-running".format(
                    clone["id"], entry["name"], project_id,
                    sorted(have), sorted(want),
                )
            )
        found[volume_id] = clone
    return found


def summarise(volumes, done, only_files):
    """Print the per-volume plan."""
    print("\n{:>7} {:>7} {:<42} {:>6} {:>7} {}".format(
        "volume", "frames", "name", "clone", "drop", "videos"))
    for volume_id, entry in sorted(volumes.items()):
        state = "have" if volume_id in done else "clone"
        keep = sorted(entry["videos"])
        drop = len(entry["dropped"]) if only_files else 0
        print("{:>7} {:>7} {:<42} {:>6} {:>7} {}".format(
            volume_id, entry["frames"], entry["name"][:42], state, drop,
            ",".join(str(v) for v in keep)))
    files = sum(len(e["videos"]) if only_files else len(e["all_files"])
                for e in volumes.values())
    print("\n{} volume(s), {} of them already cloned; {} video file(s) in total".format(
        len(volumes), len(done), files))


def verify_clone(api, clone_id, entry, only_files):
    """Check a fresh clone holds the right files and no annotations.

    Raises:
        SystemExit: On a file-set mismatch or a surviving annotation.
    """
    want = set(entry["videos"].values()) if only_files else None
    expected = len(entry["videos"]) if only_files else len(entry["all_files"])
    filenames = wait_for_files(api, clone_id, expected)
    if want is not None and set(filenames.values()) != want:
        raise SystemExit(
            "clone {} holds {} but should hold {}".format(
                clone_id, sorted(filenames.values()), sorted(want))
        )
    leftover = {}
    for file_id in filenames:
        annotations = api.video_annotations(int(file_id))
        if annotations:
            leftover[int(file_id)] = len(annotations)
    if leftover:
        raise SystemExit(
            "clone {} came out with annotations ({}) -- clone_annotations was "
            "not honoured, delete the volume before re-running".format(
                clone_id, leftover)
        )
    return sorted(int(f) for f in filenames)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--project", type=int, default=DEFAULT_PROJECT,
                        help="target project; needs project admin")
    parser.add_argument("--all-files", action="store_true",
                        help="clone whole volumes instead of only the split's videos")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N volumes, for a trial run")
    parser.add_argument("--commit", action="store_true",
                        help="actually clone; without it nothing is written to BIIGLE")
    parser.add_argument("--email", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    only_files = not args.all_files

    api = BiigleApi(email=args.email, token=args.token)
    user = api.whoami()
    print("authenticated as {} {} <{}>".format(
        user.get("firstname"), user.get("lastname"), user.get("email")))

    video_counts = scan_video_ids(args.dataset, args.split)
    volumes = resolve_volumes(api, video_counts)
    if args.limit:
        keep = sorted(volumes)[:args.limit]
        volumes = {k: volumes[k] for k in keep}
        print("--limit {}: {} volume(s)".format(args.limit, len(volumes)))

    done = existing_clones(api, args.project, volumes)
    summarise(volumes, done, only_files)

    os.makedirs(args.out, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "project_id": args.project,
        "only_files": only_files,
        "frames": sum(video_counts.values()),
        "videos": len(video_counts),
        "volumes": {},
    }
    for volume_id, entry in sorted(volumes.items()):
        clone = done.get(volume_id)
        report["volumes"][str(volume_id)] = {
            "name": entry["name"],
            "url": entry["url"],
            "source_projects": entry["projects"],
            "videos": {str(v): n for v, n in sorted(entry["videos"].items())},
            "frames": entry["frames"],
            "dropped_files": entry["dropped"] if only_files else [],
            "clone_id": clone["id"] if clone else None,
            "cloned_now": False,
        }

    pending = [v for v in sorted(volumes) if v not in done]
    if not args.commit:
        print("\ndry run -- nothing cloned. Re-run with --commit to clone "
              "{} volume(s).".format(len(pending)))
        return
    if not pending:
        print("\nnothing left to clone")
    else:
        print("\ncloning {} volume(s) into project {}".format(len(pending), args.project))

    for index, volume_id in enumerate(pending, 1):
        entry = volumes[volume_id]
        clone = api.clone_volume(
            volume_id,
            args.project,
            only_files=sorted(entry["videos"]) if only_files else None,
            clone_annotations=False,
            clone_file_labels=False,
        )
        clone_id = clone["id"]
        files = verify_clone(api, clone_id, entry, only_files)
        record = report["volumes"][str(volume_id)]
        record["clone_id"] = clone_id
        record["cloned_now"] = True
        record["clone_files"] = files
        print("  [{}/{}] {} -> {} ({}), {} file(s), no annotations".format(
            index, len(pending), volume_id, clone_id, entry["name"], len(files)))

    report_path = os.path.join(args.out, "clone_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote {}".format(report_path))


if __name__ == "__main__":
    main()
