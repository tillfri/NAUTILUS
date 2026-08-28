"""Split the single refined-test BIIGLE volume into one volume per station video.

``upload_refined_to_biigle.py`` put the whole refined **test** split -- 1303 frames,
2499 pre-drawn boxes -- into one volume, **35740 ``thuenen_refined_test``** in project
**5280 ``Thuenen Refine Images``**. One volume cannot be divided between annotators:
there is no way to say "you take these frames, I take those", so two reviewers either
collide or both start at the top of the same list. This script cuts it into 19 volumes,
one per source station video, so a reviewer can claim a station and work it end to end --
and so each image volume pairs 1:1 with the station's video volume that
``clone_test_video_volumes.py`` already put in project **5293 ``Thuenen Refine Video``**.

Nothing is re-uploaded, and that is the point:

* **The pixels never move.** Volume 35740's ``url`` is ``user-2576://thuenen``, our own
  BIIGLE storage disk, and a volume is only ``(url, file list)``. Subsets of it need no
  file transfer at all.
* **``POST volumes/:id/clone-to/:project_id`` takes ``only_files`` *and*
  ``clone_annotations``.** That combination is BIIGLE's only volume-split primitive --
  there is no move/split endpoint and no ``DELETE volumes/:id``.
* **Re-posting the boxes would be lossy.** Neither annotation-create endpoint accepts a
  creator field, so every box would come back owned by the caller. Review has already
  started (12 boxes created by another user, 3 moved, 22 machine boxes deleted, all in
  the AB08/AB12 stations), and cloning copies the database rows -- ``user_id``,
  ``confidence`` and geometry survive. Verification therefore checks ``user_id``
  explicitly: if it comes back as the caller everywhere, ``clone_annotations``
  re-attributed and the run must stop with the original still intact.

Three things shape the script:

* **Snapshot first, always, even on a dry run.** ``snapshot_annotations.json`` is the only
  copy of the reviewers' corrections that exists outside BIIGLE, so nothing is written
  before it is on disk. A timestamped copy is kept under ``snapshots/`` as well.
* **The target project is the state.** ``clone-to`` has no upsert -- posting the same
  clone twice makes two volumes. Each station is skipped when project 5280 already holds
  a volume of the computed name whose filename set matches, so re-running is a complete
  resume. Same discipline as ``clone_test_video_volumes.py``.
* **The old volume is archived, not deleted.** With ``--archive``, 35740 is attached to a
  second project and then detached from 5280 *without* ``force``: it leaves the reviewers'
  project but survives whole as the rollback. ``BiigleApi.detach_project_volume`` does not
  expose ``force`` at all, so this script cannot destroy it.

Image IDs change: ``runs/biigle_upload_test/upload_payload.json`` and ``upload_state.json``
go stale the moment 35740 is retired. ``split_report.json`` records the old->new image ID
map per station; from here on the **filename** is the only stable identifier.

Usage:
    # dry run: snapshot, resolve the 19 stations, print the plan. Writes nothing to BIIGLE.
    docker exec nautilus-qwen python3 /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/split_test_volume_by_station.py \
        --source-volume 35740 --project 5280 --out /workspace/runs/biigle_split_test

    # trial: the smallest station only (MGF_SAR_Hol10_St434_AB, 2 frames), then look at
    # it in the BIIGLE UI before doing the rest
    ... split_test_volume_by_station.py ... --limit 1 --commit

    # the other 18; re-running clones 0 and skips 19
    ... split_test_volume_by_station.py ... --commit

    # only once all 19 verify: 35740 leaves project 5280 for an archive project
    ... split_test_volume_by_station.py ... --commit --archive
"""

import argparse
import collections
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from biigle_api import BiigleApi  # noqa: E402
from clone_test_video_volumes import VIDEO_ID_RE, filenames_map, wait_for_files  # noqa: E402

DEFAULT_OUT = "/workspace/runs/biigle_split_test"
# thuenen_refined_test, the single volume upload_refined_to_biigle.py wrote.
DEFAULT_SOURCE_VOLUME = 35740
# Thuenen Refine Images, the reviewers' project.
DEFAULT_PROJECT = 5280
DEFAULT_ARCHIVE_NAME = "Thuenen Refine Archive"


def snapshot(api, volume_id, out_dir):
    """Write the source volume's full state to disk before anything is touched.

    Returns:
        ``(volume, {file_id: filename}, [annotation, ...])``.
    """
    volume = api.volume_info(volume_id)
    filenames = filenames_map(api, volume_id)
    annotations = api.volume_annotations(volume_id)
    if not filenames:
        raise SystemExit("volume {} reports no files".format(volume_id))

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    keep_dir = os.path.join(out_dir, "snapshots", stamp)
    os.makedirs(keep_dir, exist_ok=True)
    for name, payload in (("snapshot_volume.json", volume),
                          ("snapshot_filenames.json", filenames),
                          ("snapshot_annotations.json", annotations)):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        shutil.copyfile(path, os.path.join(keep_dir, name))
    print("snapshot: {} image(s), {} annotation(s) -> {} (kept as {})".format(
        len(filenames), len(annotations), out_dir, os.path.join("snapshots", stamp)))
    return volume, filenames, annotations


def annotation_key(annotation):
    """Reduce an annotation to what a clone must reproduce exactly.

    Geometry plus, per label, ``(label_id, user_id, confidence)``. The row IDs
    themselves are new in the clone and carry no information; ``user_id`` is the
    field the whole approach stands on.
    """
    points = tuple(round(float(p), 6) for p in annotation["points"])
    labels = tuple(sorted(
        (label["label_id"], label["user_id"], round(float(label["confidence"]), 6))
        for label in annotation.get("labels", [])
    ))
    return annotation["shape_id"], points, labels


def annotations_by_filename(annotations, filenames):
    """Group annotations into ``{filename: Counter(annotation_key)}``.

    Raises:
        SystemExit: If an annotation belongs to an image the volume does not list.
    """
    grouped = collections.defaultdict(collections.Counter)
    for annotation in annotations:
        name = filenames.get(str(annotation["image_id"]))
        if name is None:
            raise SystemExit(
                "annotation {} references image {}, which is not in the volume's "
                "filename map".format(annotation["id"], annotation["image_id"])
            )
        grouped[name][annotation_key(annotation)] += 1
    return grouped


def label_user_histogram(annotations):
    """Return ``{user_id: label count}`` -- the attribution fingerprint."""
    counts = collections.Counter()
    for annotation in annotations:
        for label in annotation.get("labels", []):
            counts[label["user_id"]] += 1
    return dict(counts)


def group_by_station(api, filenames, ann_by_filename):
    """Group the volume's files by the source *volume* of the video they came from.

    ``build_dataset.py`` names a frame ``<video stem>(<video id>)_f<frame>.jpg``; the
    parenthesised group resolves through ``GET videos/:id`` to the station volume. The
    mapping is many-to-one -- 38329 and 38330 are two videos of the same station -- which
    is why 20 videos become 19 volumes.

    Returns:
        ``{station_volume_id: {"name", "source_name", "videos", "image_ids",
        "filenames", "annotations"}}``.

    Raises:
        SystemExit: If a filename carries no video ID.
    """
    videos = collections.defaultdict(list)
    for file_id, name in sorted(filenames.items(), key=lambda kv: kv[1]):
        matches = VIDEO_ID_RE.findall(name)
        if not matches:
            raise SystemExit("frame {!r} carries no (<video id>) group".format(name))
        videos[int(matches[-1])].append((int(file_id), name))

    stations = {}
    for video_id in sorted(videos):
        volume_id = api.video_info(video_id)["volume_id"]
        entry = stations.get(volume_id)
        if entry is None:
            entry = stations[volume_id] = {
                "source_name": api.volume_info(volume_id)["name"],
                "videos": [],
                "image_ids": [],
                "filenames": [],
            }
        entry["videos"].append(video_id)
        for file_id, name in videos[video_id]:
            entry["image_ids"].append(file_id)
            entry["filenames"].append(name)

    for entry in stations.values():
        entry["image_ids"].sort()
        entry["filenames"].sort()
        entry["annotations"] = sum(
            sum(ann_by_filename.get(n, {}).values()) for n in entry["filenames"])
        # The frame count goes in the name so a reviewer can see the size of a
        # job before claiming it.
        entry["name"] = "{} ({})".format(entry["source_name"], len(entry["filenames"]))

    check_partition(stations, filenames)
    return stations


def check_partition(stations, filenames):
    """Assert the stations cover every file exactly once, and names are unique.

    Raises:
        SystemExit: On an overlap, a leftover, or a duplicated volume name.
    """
    seen = collections.Counter()
    for entry in stations.values():
        seen.update(entry["image_ids"])
    duplicated = [i for i, n in seen.items() if n > 1]
    if duplicated:
        raise SystemExit("image(s) assigned to more than one station: {}".format(
            sorted(duplicated)[:10]))
    missing = set(int(i) for i in filenames) - set(seen)
    if missing:
        raise SystemExit("{} image(s) assigned to no station: {}".format(
            len(missing), sorted(missing)[:10]))
    names = collections.Counter(e["name"] for e in stations.values())
    clashes = [n for n, c in names.items() if c > 1]
    if clashes:
        raise SystemExit("station name(s) not unique: {}".format(clashes))


def order_stations(stations, limit):
    """Return the station IDs to act on, smallest first.

    Smallest first so ``--limit 1`` is the cheapest possible probe of the three
    things the API docs do not answer: whether ``clone-to`` accepts the source's
    own project, whether ``clone_annotations`` keeps ``user_id``, and whether the
    clone's images actually render.
    """
    order = sorted(stations, key=lambda v: (len(stations[v]["filenames"]), v))
    return order[:limit] if limit else order


def existing_clones(api, project_id, stations, source_volume):
    """Return ``{station_volume_id: clone}`` for clones already in the project.

    Matched by the computed name *and* the filename set: a same-named volume with
    a different file set is a half-made clone, and saying so is more useful than
    silently making a second one.

    Raises:
        SystemExit: On a duplicated name or a file-set mismatch.
    """
    by_name = {}
    for volume in api.project_volumes(project_id):
        if volume["id"] == source_volume:
            continue
        by_name.setdefault(volume["name"], []).append(volume)
    found = {}
    for volume_id, entry in sorted(stations.items()):
        candidates = by_name.get(entry["name"], [])
        if not candidates:
            continue
        if len(candidates) > 1:
            raise SystemExit(
                "project {} already holds {} volumes named {!r} -- a duplicated "
                "clone; delete the extras before re-running".format(
                    project_id, len(candidates), entry["name"])
            )
        clone = candidates[0]
        have = set(filenames_map(api, clone["id"]).values())
        want = set(entry["filenames"])
        if have != want:
            raise SystemExit(
                "volume {} ({!r}) in project {} holds {} file(s) instead of the "
                "expected {} -- resolve it by hand before re-running".format(
                    clone["id"], entry["name"], project_id, len(have), len(want))
            )
        found[volume_id] = clone
    return found


def verify_clone(api, clone_id, entry, ann_by_filename):
    """Check a clone against the snapshot, box for box.

    Filename set first, then per filename the *multiset* of ``annotation_key`` --
    so a moved box, a dropped box, a re-attributed box or a changed confidence all
    fail. These are database copies; there is no tolerance to allow.

    Returns:
        ``{filename: new_image_id}``.

    Raises:
        SystemExit: On any mismatch. The clone is left in place for inspection.
    """
    filenames = wait_for_files(api, clone_id, len(entry["filenames"]))
    if set(filenames.values()) != set(entry["filenames"]):
        raise SystemExit(
            "clone {} holds {} file(s) but should hold {}".format(
                clone_id, len(filenames), len(entry["filenames"]))
        )
    got = annotations_by_filename(api.volume_annotations(clone_id), filenames)
    problems = []
    for name in entry["filenames"]:
        want = ann_by_filename.get(name, collections.Counter())
        have = got.get(name, collections.Counter())
        if want == have:
            continue
        detail = ("same count, different content" if sum(want.values()) == sum(have.values())
                  else "expected {}, found {}".format(sum(want.values()), sum(have.values())))
        problems.append("{}: {}".format(name, detail))
    if problems:
        raise SystemExit(
            "clone {} ({!r}) does not match the snapshot in {} image(s):\n  {}".format(
                clone_id, entry["name"], len(problems), "\n  ".join(problems[:10]))
        )
    return {name: int(file_id) for file_id, name in filenames.items()}


def summarise(stations, order, done):
    """Print the per-station plan, largest station first."""
    print("\n{:>7} {:>6} {:>5} {:<44} {:>6} {}".format(
        "volume", "frames", "anns", "new volume name", "state", "videos"))
    for volume_id in sorted(order, key=lambda v: -len(stations[v]["filenames"])):
        entry = stations[volume_id]
        print("{:>7} {:>6} {:>5} {:<44} {:>6} {}".format(
            volume_id, len(entry["filenames"]), entry["annotations"], entry["name"][:44],
            "have" if volume_id in done else "clone",
            ",".join(str(v) for v in entry["videos"])))
    frames = sum(len(stations[v]["filenames"]) for v in order)
    anns = sum(stations[v]["annotations"] for v in order)
    print("\n{} station(s), {} of them already cloned; {} frame(s), {} annotation(s)".format(
        len(order), sum(1 for v in order if v in done), frames, anns))


def archive_source(api, source_volume, project_id, archive_project, archive_name):
    """Move the source volume out of the reviewers' project without destroying it.

    Attach to a second project first, so that detaching from ``project_id`` cannot
    be a deletion: BIIGLE only destroys a volume when the project it is detached
    from is its last one, and ``detach_project_volume`` never sends ``force``.

    Returns:
        The archive record for the report.
    """
    if not archive_project:
        project = api.create_project(
            archive_name,
            description="Retired volumes from the Thuenen refined-test review. "
                        "Rollback only -- do not annotate here.")
        archive_project = project["id"]
        print("created archive project {} ({!r})".format(archive_project, archive_name))

    before = [p["id"] for p in api.volume_info(source_volume).get("projects", [])]
    if archive_project not in before:
        api.attach_project_volume(archive_project, source_volume)
        print("attached volume {} to project {}".format(source_volume, archive_project))

    projects = [p["id"] for p in api.volume_info(source_volume).get("projects", [])]
    if archive_project not in projects:
        raise SystemExit(
            "volume {} did not attach to project {} (projects: {}) -- refusing to "
            "detach, that would delete it".format(source_volume, archive_project, projects))
    if project_id not in projects:
        print("volume {} is already detached from project {}".format(source_volume, project_id))
    elif len(projects) < 2:
        raise SystemExit(
            "volume {} belongs to {} project(s); detaching would delete it".format(
                source_volume, len(projects)))
    else:
        api.detach_project_volume(project_id, source_volume)
        print("detached volume {} from project {}".format(source_volume, project_id))

    after = [p["id"] for p in api.volume_info(source_volume).get("projects", [])]
    if project_id in after or archive_project not in after:
        raise SystemExit(
            "volume {} ended up in projects {} -- expected {} gone and {} present".format(
                source_volume, after, project_id, archive_project))
    print("volume {} survives in project(s) {}".format(source_volume, after))
    return {"archive_project_id": archive_project, "projects_before": before,
            "projects_after": after}


def finish_station(api, record, entry, ann_by_filename, old_ids):
    """Verify one clone and fill in its report record (new IDs, old->new map)."""
    filename_map = verify_clone(api, record["clone_id"], entry, ann_by_filename)
    record["verified"] = True
    record["filename_map"] = filename_map
    record["image_id_map"] = {str(old_ids[n]): filename_map[n] for n in entry["filenames"]}
    record["annotations_found"] = entry["annotations"]


def write_report(out_dir, report):
    """Write ``split_report.json`` -- including the old->new image ID map."""
    path = os.path.join(out_dir, "split_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-volume", type=int, default=DEFAULT_SOURCE_VOLUME)
    parser.add_argument("--project", type=int, default=DEFAULT_PROJECT,
                        help="project holding the source volume; the clones land here too")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the N smallest stations, for a trial run")
    parser.add_argument("--commit", action="store_true",
                        help="actually clone; without it nothing is written to BIIGLE")
    parser.add_argument("--archive", action="store_true",
                        help="after all stations verify, move the source volume out of "
                             "--project into an archive project (never deletes it)")
    parser.add_argument("--archive-project", type=int, default=0,
                        help="existing archive project; 0 creates one")
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    parser.add_argument("--no-verify-existing", action="store_true",
                        help="skip the box-for-box re-verification of clones an earlier "
                             "run made -- needed once reviewers have started editing them, "
                             "since their edits legitimately diverge from the snapshot")
    parser.add_argument("--email", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    api = BiigleApi(email=args.email, token=args.token)
    user = api.whoami()
    print("authenticated as {} {} <{}>".format(
        user.get("firstname"), user.get("lastname"), user.get("email")))

    os.makedirs(args.out, exist_ok=True)
    volume, filenames, annotations = snapshot(api, args.source_volume, args.out)
    source_projects = [p["id"] for p in volume.get("projects", [])]
    print("source volume {!r} ({}), url {}, project(s) {}".format(
        volume.get("name"), args.source_volume, volume.get("url"), source_projects))

    ann_by_filename = annotations_by_filename(annotations, filenames)
    histogram = label_user_histogram(annotations)
    print("label attribution in the snapshot: {}".format(
        {str(k): v for k, v in sorted(histogram.items())}))

    stations = group_by_station(api, filenames, ann_by_filename)
    order = order_stations(stations, args.limit)
    if args.limit:
        print("--limit {}: {} of {} station(s)".format(args.limit, len(order), len(stations)))

    selected = {v: stations[v] for v in order}
    done = existing_clones(api, args.project, selected, args.source_volume)
    summarise(stations, order, done)

    report = {
        "source_volume": args.source_volume,
        "source_volume_name": volume.get("name"),
        "source_volume_url": volume.get("url"),
        "project_id": args.project,
        "snapshot": {
            "images": len(filenames),
            "annotations": len(annotations),
            "label_user_ids": {str(k): v for k, v in sorted(histogram.items())},
        },
        "stations": {},
        "archive": None,
    }
    for volume_id in sorted(stations):
        entry = stations[volume_id]
        clone = done.get(volume_id)
        report["stations"][str(volume_id)] = {
            "name": entry["name"],
            "source_volume_name": entry["source_name"],
            "videos": entry["videos"],
            "frames": len(entry["filenames"]),
            "annotations_expected": entry["annotations"],
            "annotations_found": None,
            "in_this_run": volume_id in selected,
            "clone_id": clone["id"] if clone else None,
            "cloned_now": False,
            "verified": False,
            "filename_map": {},
            "image_id_map": {},
        }

    pending = [v for v in order if v not in done]
    if not args.commit:
        print("\ndry run -- nothing written to BIIGLE. Re-run with --commit to clone "
              "{} station(s).".format(len(pending)))
        if args.archive:
            print("--archive is a no-op without --commit.")
        write_report(args.out, report)
        return

    if pending:
        print("\ncloning {} station(s) into project {}".format(len(pending), args.project))
    else:
        print("\nnothing left to clone")

    old_ids = {name: int(file_id) for file_id, name in filenames.items()}
    for index, volume_id in enumerate(pending, 1):
        entry = stations[volume_id]
        try:
            clone = api.clone_volume(
                args.source_volume,
                args.project,
                name=entry["name"],
                only_files=entry["image_ids"],
                clone_annotations=True,
                clone_file_labels=True,
            )
        except Exception as error:
            raise SystemExit(
                "clone-to refused {!r} into project {}: {}\nIf the source volume's own "
                "project is not accepted, clone into a staging project instead and move "
                "the result with attach_project_volume/detach_project_volume.".format(
                    entry["name"], args.project, error)
            )
        done[volume_id] = clone
        record = report["stations"][str(volume_id)]
        record["clone_id"] = clone["id"]
        record["cloned_now"] = True
        finish_station(api, record, entry, ann_by_filename, old_ids)
        print("  [{}/{}] {} -> {} ({!r}), {} frame(s), {} annotation(s) verified".format(
            index, len(pending), volume_id, clone["id"], entry["name"],
            len(entry["filenames"]), entry["annotations"]))

    if not args.no_verify_existing:
        stale = [v for v in order if not report["stations"][str(v)]["cloned_now"]]
        if stale:
            print("re-verifying {} clone(s) from an earlier run".format(len(stale)))
        for volume_id in stale:
            record = report["stations"][str(volume_id)]
            record["clone_id"] = done[volume_id]["id"]
            finish_station(api, record, stations[volume_id], ann_by_filename, old_ids)

    verified = [v for v in order if report["stations"][str(v)]["verified"]]
    total = sum(report["stations"][str(v)]["annotations_found"] for v in verified)
    print("\n{}/{} station(s) verified, {} annotation(s) accounted for".format(
        len(verified), len(order), total))

    if args.archive:
        missing = [v for v in stations if not report["stations"][str(v)]["verified"]]
        if missing:
            raise SystemExit(
                "refusing to archive: {} station(s) not verified in this run ({}). "
                "Run without --limit and without --no-verify-existing first.".format(
                    len(missing), sorted(missing)))
        if total != len(annotations):
            raise SystemExit(
                "refusing to archive: the clones hold {} annotation(s), the snapshot "
                "{}".format(total, len(annotations)))
        report["archive"] = archive_source(
            api, args.source_volume, args.project, args.archive_project, args.archive_name)

    write_report(args.out, report)


if __name__ == "__main__":
    main()
