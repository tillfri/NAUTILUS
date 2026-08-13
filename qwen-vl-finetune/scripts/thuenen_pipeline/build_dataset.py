"""Extract annotated frames from the BIIGLE videos and write a YOLO dataset.

Layout produced under ``--out-dir`` (default ``.../datasets/thuenen``)::

    thuenen/
      classes.txt                 global class list, index == YOLO class id
      class_map.csv               class_id, label_name, dir_name, box counts
      rejected_annotations.csv    every dropped row / keyframe / box, with reason
      dataset_report.json         summary counters
      <class_dir>/
        images/<stem>(<video_id>)_f<frame>.jpg
        labels/<stem>(<video_id>)_f<frame>.txt

All annotations that fall on the same ``(video, frame index)`` share one frame
image. A frame is written into the directory of every class annotated on it,
and the label file inside a class directory lists only that class's boxes --
so each class directory is a usable single-class dataset while class ids stay
globally consistent (see ``classes.txt``) for merging later.

Timestamps in the reports are seconds; the frame index is ``floor(t * fps)``
with fps read from the video. Boxes are normalised against the *decoded* frame
size, not the size recorded in the report -- two videos claim a width of 1921
in their metadata, which would skew the normalisation.

Usage:
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/build_dataset.py --videos-dir /workspace/datasets/biigle_videos \
--out-dir /workspace/datasets/thuenen
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotations import build_class_index, parse_reports, video_stem  # noqa: E402

DEFAULT_REPORTS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "biigle_reports")
)
DEFAULT_VIDEOS = "/home/tfricke/nautilus/datasets/biigle_videos"
DEFAULT_OUT = "/home/tfricke/nautilus/datasets/thuenen"

# Re-seeking costs a keyframe-to-target decode; decoding forward is cheaper
# when the next wanted frame is close by.
FORWARD_DECODE_WINDOW_S = 3.0


class FrameReader(object):
    """Pull specific frames out of a video by timestamp, in ascending order."""

    def __init__(self, path):
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        rate = self.stream.average_rate or self.stream.guessed_rate
        if not rate:
            raise ValueError("no frame rate for {}".format(path))
        self.fps = float(rate)
        self.time_base = float(self.stream.time_base)
        self.start_time = (self.stream.start_time or 0) * self.time_base
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.duration = float(self.stream.duration * self.time_base) if self.stream.duration else None
        self._decoder = None
        self._position = None

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _rewind(self):
        """Reopen the container so decoding restarts from the first frame."""
        self.container.close()
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def _seek(self, target):
        offset = int((target + self.start_time) / self.time_base)
        try:
            self.container.seek(offset, stream=self.stream, backward=True, any_frame=False)
        except av.FFmpegError:
            # Several of these MP4s reject a seek to (or near) offset 0 with
            # EPERM. Decoding from the head reaches the same frame, and this
            # only ever happens near the start, so it stays cheap.
            self._rewind()
        self._decoder = self.container.decode(self.stream)
        self._position = None

    def frame_at(self, target):
        """Return the frame displayed at ``target`` seconds as a BGR ndarray.

        Args:
            target: Presentation time in seconds. Must not go backwards
                between calls by more than a seek allows.

        Returns:
            ndarray in BGR order, or ``None`` if the video ended first.
        """
        if (self._decoder is None or self._position is None
                or target < self._position
                or target > self._position + FORWARD_DECODE_WINDOW_S):
            self._seek(target)

        # A frame shown at `target` starts at or before it; accept the first
        # frame whose own start is within half a frame of the target.
        tolerance = 0.5 / self.fps
        for frame in self._decoder:
            if frame.pts is None:
                continue
            timestamp = frame.pts * self.time_base - self.start_time
            self._position = timestamp
            if timestamp >= target - tolerance:
                return frame.to_ndarray(format="bgr24")
        return None


def find_video(videos_dir, video_id, video_filename):
    """Locate a downloaded video, preferring the ``<id>_<filename>`` name."""
    exact = os.path.join(videos_dir, "{}_{}".format(video_id, video_filename))
    if os.path.isfile(exact):
        return exact
    matches = [p for p in glob.glob(os.path.join(videos_dir, "{}_*".format(video_id)))
               if not p.endswith(".part")]
    return matches[0] if matches else None


def clip_box(annotation, width, height, min_box_px, max_area_frac):
    """Clip a box to the frame and decide whether it survives.

    Args:
        annotation: A ``BoxAnnotation``.
        width: Decoded frame width.
        height: Decoded frame height.
        min_box_px: Minimum side length in pixels after clipping.
        max_area_frac: Reject boxes covering more than this fraction of the frame.

    Returns:
        ``(box, None)`` with ``box = (x1, y1, x2, y2)``, or ``(None, reason)``.
    """
    if annotation.x2 <= 0 or annotation.y2 <= 0 or annotation.x1 >= width or annotation.y1 >= height:
        return None, "outside_frame"

    x1 = max(0.0, min(float(annotation.x1), width))
    y1 = max(0.0, min(float(annotation.y1), height))
    x2 = max(0.0, min(float(annotation.x2), width))
    y2 = max(0.0, min(float(annotation.y2), height))

    if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
        return None, "too_small_after_clip"
    if (x2 - x1) * (y2 - y1) > max_area_frac * width * height:
        return None, "covers_whole_frame"
    return (x1, y1, x2, y2), None


def to_yolo(box, width, height):
    """Convert a pixel box to normalised YOLO ``cx cy w h``, clamped to [0, 1]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)


def write_outputs(out_dir, class_dirs, image_name, jpeg_bytes, boxes_by_class):
    """Write the frame image and one label file per class present on it."""
    for class_id, lines in boxes_by_class.items():
        class_root = os.path.join(out_dir, class_dirs[class_id])
        images_dir = os.path.join(class_root, "images")
        labels_dir = os.path.join(class_root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        with open(os.path.join(images_dir, image_name + ".jpg"), "wb") as handle:
            handle.write(jpeg_bytes)
        with open(os.path.join(labels_dir, image_name + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-box-px", type=float, default=4.0,
                        help="Drop boxes thinner than this after clipping")
    parser.add_argument("--max-area-frac", type=float, default=0.9,
                        help="Drop boxes covering more than this fraction of the frame")
    parser.add_argument("--only-video", type=int, action="append", default=None,
                        help="Restrict to these video IDs (repeatable); useful for testing")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="Process at most N videos")
    args = parser.parse_args()

    annotations, rejects, row_count = parse_reports(args.reports_dir)
    classes = build_class_index(annotations)
    class_id_by_label = {c["label_name"]: c["class_id"] for c in classes}
    class_dirs = {c["class_id"]: c["dir_name"] for c in classes}
    print("{} CSV rows -> {} boxes over {} classes; {} entries rejected while parsing".format(
        row_count, len(annotations), len(classes), len(rejects)))

    by_video = defaultdict(list)
    for annotation in annotations:
        by_video[annotation.video_id].append(annotation)

    video_ids = sorted(by_video)
    if args.only_video:
        video_ids = [v for v in video_ids if v in set(args.only_video)]
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]

    os.makedirs(args.out_dir, exist_ok=True)
    stats = Counter()
    boxes_per_class = Counter()
    images_per_class = Counter()
    resolution_mismatches = []
    started = time.time()

    for position, video_id in enumerate(video_ids, 1):
        video_annotations = by_video[video_id]
        filename = video_annotations[0].video_filename
        path = find_video(args.videos_dir, video_id, filename)
        if path is None:
            print("[{}/{}] {} MISSING video file, skipping {} boxes".format(
                position, len(video_ids), video_id, len(video_annotations)))
            stats["videos_missing"] += 1
            for annotation in video_annotations:
                rejects.append({
                    "source_csv": annotation.source_csv, "video_id": video_id,
                    "video_filename": filename, "label_name": annotation.label_name,
                    "shape_name": annotation.shape_name, "annotation_id": annotation.annotation_id,
                    "keyframe": annotation.keyframe, "reason": "video_not_downloaded", "detail": "",
                    "video_annotation_label_id": "",
                })
            continue

        try:
            reader = FrameReader(path)
        except Exception as error:  # noqa: BLE001
            print("[{}/{}] {} UNREADABLE ({}), skipping".format(
                position, len(video_ids), video_id, error))
            stats["videos_unreadable"] += 1
            continue

        with reader:
            width, height = reader.width, reader.height
            meta = (video_annotations[0].meta_width, video_annotations[0].meta_height)
            if meta != (0, 0) and meta != (width, height):
                resolution_mismatches.append({
                    "video_id": video_id, "video_filename": filename,
                    "report": "{}x{}".format(*meta), "decoded": "{}x{}".format(width, height),
                })

            # Group annotations onto the frame that is on screen at their timestamp.
            frames = defaultdict(list)
            for annotation in video_annotations:
                frame_index = int(annotation.timestamp * reader.fps)
                frames[frame_index].append(annotation)

            written = 0
            for frame_index in sorted(frames):
                target = frame_index / reader.fps
                try:
                    image = reader.frame_at(target)
                except Exception as error:  # noqa: BLE001
                    image = None
                    stats["decode_errors"] += 1
                if image is None:
                    stats["frames_unavailable"] += 1
                    for annotation in frames[frame_index]:
                        rejects.append({
                            "source_csv": annotation.source_csv, "video_id": video_id,
                            "video_filename": filename, "label_name": annotation.label_name,
                            "shape_name": annotation.shape_name,
                            "annotation_id": annotation.annotation_id,
                            "keyframe": annotation.keyframe, "reason": "frame_not_decodable",
                            "detail": "t={:.3f}s idx={}".format(target, frame_index),
                            "video_annotation_label_id": "",
                        })
                    continue

                boxes_by_class = defaultdict(list)
                for annotation in frames[frame_index]:
                    box, reason = clip_box(annotation, width, height,
                                           args.min_box_px, args.max_area_frac)
                    if box is None:
                        stats["boxes_rejected"] += 1
                        rejects.append({
                            "source_csv": annotation.source_csv, "video_id": video_id,
                            "video_filename": filename, "label_name": annotation.label_name,
                            "shape_name": annotation.shape_name,
                            "annotation_id": annotation.annotation_id,
                            "keyframe": annotation.keyframe, "reason": reason,
                            "detail": "({:.1f},{:.1f},{:.1f},{:.1f}) in {}x{}".format(
                                annotation.x1, annotation.y1, annotation.x2, annotation.y2,
                                width, height),
                            "video_annotation_label_id": "",
                        })
                        continue
                    class_id = class_id_by_label[annotation.label_name]
                    cx, cy, bw, bh = to_yolo(box, width, height)
                    boxes_by_class[class_id].append(
                        "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, cx, cy, bw, bh))
                    stats["boxes_written"] += 1
                    boxes_per_class[class_id] += 1

                if not boxes_by_class:
                    stats["frames_without_valid_boxes"] += 1
                    continue

                ok, buffer = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if not ok:
                    stats["encode_errors"] += 1
                    continue
                image_name = "{}({})_f{:06d}".format(video_stem(filename), video_id, frame_index)
                write_outputs(args.out_dir, class_dirs, image_name,
                              np.asarray(buffer).tobytes(), boxes_by_class)
                for class_id in boxes_by_class:
                    images_per_class[class_id] += 1
                stats["images_written"] += len(boxes_by_class)
                stats["frames_extracted"] += 1
                written += 1

            stats["videos_processed"] += 1
            elapsed = time.time() - started
            print("[{}/{}] {} {} -- {} frames, {:.1f} fps src, {}x{} ({:.1f} min elapsed)".format(
                position, len(video_ids), video_id, filename, written,
                reader.fps, width, height, elapsed / 60.0))

    # Reports -------------------------------------------------------------
    with open(os.path.join(args.out_dir, "classes.txt"), "w", encoding="utf-8") as handle:
        for entry in classes:
            handle.write(entry["label_name"] + "\n")

    with open(os.path.join(args.out_dir, "class_map.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "label_name", "dir_name", "annotations_parsed",
                         "boxes_written", "images_written"])
        for entry in classes:
            writer.writerow([entry["class_id"], entry["label_name"], entry["dir_name"],
                             entry["count"], boxes_per_class[entry["class_id"]],
                             images_per_class[entry["class_id"]]])

    reject_fields = ["source_csv", "video_annotation_label_id", "annotation_id", "video_id",
                     "video_filename", "label_name", "shape_name", "keyframe", "reason", "detail"]
    with open(os.path.join(args.out_dir, "rejected_annotations.csv"), "w",
              encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reject_fields, extrasaction="ignore")
        writer.writeheader()
        for entry in rejects:
            writer.writerow(entry)

    report = {
        "csv_rows": row_count,
        "boxes_parsed": len(annotations),
        "classes": len(classes),
        "videos_selected": len(video_ids),
        "reject_reasons": dict(Counter(entry["reason"] for entry in rejects)),
        "resolution_mismatches": resolution_mismatches,
        "runtime_minutes": round((time.time() - started) / 60.0, 2),
        "settings": {
            "jpeg_quality": args.jpeg_quality,
            "min_box_px": args.min_box_px,
            "max_area_frac": args.max_area_frac,
        },
    }
    report.update(stats)
    with open(os.path.join(args.out_dir, "dataset_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\n" + json.dumps({k: v for k, v in report.items() if k != "resolution_mismatches"},
                            indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
