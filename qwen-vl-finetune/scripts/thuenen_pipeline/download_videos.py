"""Download the BIIGLE videos referenced by the annotation reports.

Every video referenced in ``biigle_reports/*/`` is fetched to
``--videos-dir`` as ``<video_id>_<original_filename>``. The expected byte size
comes from the report's ``attributes`` JSON, so completed files are skipped and
interrupted ones resume.

Usage:
    python3 thuenen_pipeline/download_videos.py
    python3 thuenen_pipeline/download_videos.py --dry-run
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotations import iter_report_files  # noqa: E402
from biigle_api import BiigleApi  # noqa: E402

DEFAULT_REPORTS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "biigle_reports")
)
DEFAULT_VIDEOS = "/home/tfricke/nautilus/datasets/biigle_videos"


def collect_videos(reports_dir):
    """Map every video referenced by the reports to its filename and byte size.

    Returns:
        Dict ``{video_id: {"filename": str, "size": int, "reports": set}}``.
    """
    videos = {}
    for path in iter_report_files(reports_dir):
        source = os.path.basename(path)
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                video_id = int(row["video_id"])
                entry = videos.setdefault(video_id, {
                    "filename": row["video_filename"],
                    "size": None,
                    "reports": set(),
                })
                entry["reports"].add(source)
                if entry["size"] is None:
                    try:
                        entry["size"] = int(json.loads(row["attributes"])["size"])
                    except (ValueError, KeyError, TypeError):
                        pass
    return videos


def local_path(videos_dir, video_id, filename):
    """Return the on-disk path for a video: ``<videos_dir>/<id>_<filename>``."""
    return os.path.join(videos_dir, "{}_{}".format(video_id, filename))


def human(num_bytes):
    """Format a byte count for progress output."""
    if num_bytes is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return "{:.1f}{}".format(num_bytes, unit)
        num_bytes /= 1024.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS)
    parser.add_argument("--email", default=None, help="Overrides BIIGLE_API_EMAIL / .env")
    parser.add_argument("--token", default=None, help="Overrides API_TOKEN / .env")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded and exit")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel downloads; a single stream only reaches ~3 MB/s")
    args = parser.parse_args()

    videos = collect_videos(args.reports_dir)
    total_size = sum(v["size"] or 0 for v in videos.values())
    print("{} videos referenced by {} reports, {} total".format(
        len(videos), len(iter_report_files(args.reports_dir)), human(total_size)))

    pending = []
    for video_id, info in sorted(videos.items()):
        path = local_path(args.videos_dir, video_id, info["filename"])
        if os.path.exists(path) and (info["size"] is None or os.path.getsize(path) == info["size"]):
            continue
        pending.append((video_id, info, path))

    print("{} already complete, {} to download ({})".format(
        len(videos) - len(pending), len(pending),
        human(sum(i["size"] or 0 for _, i, _ in pending))))

    if args.dry_run:
        for video_id, info, path in pending:
            print("  {:>6}  {:>9}  {}".format(video_id, human(info["size"]), os.path.basename(path)))
        return 0

    if not pending:
        return 0

    os.makedirs(args.videos_dir, exist_ok=True)
    api = BiigleApi(email=args.email, token=args.token)
    user = api.whoami()
    print("authenticated as {} {} <{}>".format(
        user.get("firstname"), user.get("lastname"), user.get("email")))

    # requests.Session is not thread-safe, so give every worker its own client.
    local = threading.local()

    def client():
        if not hasattr(local, "api"):
            local.api = BiigleApi(email=args.email, token=args.token)
        return local.api

    total_pending = sum(info["size"] or 0 for _, info, _ in pending)
    lock = threading.Lock()
    state = {"done_bytes": 0, "finished": 0, "last_print": 0.0}
    started = time.time()

    def report(force=False):
        now = time.time()
        if not force and now - state["last_print"] < 5.0:
            return
        state["last_print"] = now
        elapsed = max(now - started, 1e-6)
        rate = state["done_bytes"] / elapsed
        remaining = (total_pending - state["done_bytes"]) / rate / 60.0 if rate > 0 else 0
        sys.stdout.write("\r  {}/{} videos | {} / {} | {}/s | ~{:.0f} min left    ".format(
            state["finished"], len(pending), human(state["done_bytes"]),
            human(total_pending), human(rate), remaining))
        sys.stdout.flush()

    def fetch(item):
        video_id, info, path = item
        # Count bytes already on disk from an interrupted run so the ETA is honest.
        resumed = os.path.getsize(path + ".part") if os.path.exists(path + ".part") else 0
        counted = {"n": resumed}
        with lock:
            state["done_bytes"] += resumed

        def progress(done, _total, counted=counted):
            delta = done - counted["n"]
            if delta <= 0:
                return
            counted["n"] = done
            with lock:
                state["done_bytes"] += delta
                report()

        result = client().download_video(video_id, path, expected_size=info["size"],
                                         retries=args.retries, progress=progress)
        return video_id, info["filename"], result

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, item): item for item in pending}
        for future in as_completed(futures):
            video_id, info, path = futures[future]
            try:
                _, filename, result = future.result()
                message = "{} {} {}".format(video_id, os.path.basename(path), result)
            except Exception as error:  # noqa: BLE001 - keep going, report at the end
                message = "{} {} FAILED: {}".format(video_id, os.path.basename(path), error)
                failures.append((video_id, info["filename"], str(error)))
            with lock:
                state["finished"] += 1
                sys.stdout.write("\r  {}{}\n".format(message, " " * 30))
                report(force=True)

    print("\ndone in {:.1f} min, {} failed".format((time.time() - started) / 60.0, len(failures)))
    for video_id, filename, error in failures:
        print("  FAILED {} {}: {}".format(video_id, filename, error))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
