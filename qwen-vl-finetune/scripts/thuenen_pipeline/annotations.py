"""Parse BIIGLE video annotation CSV reports into flat bounding-box records.

Report schema (identical across all 49 files in ``biigle_reports/*/``)::

    video_annotation_label_id,label_id,label_name,label_hierarchy,user_id,
    firstname,lastname,video_id,video_filename,shape_id,shape_name,points,
    frames,annotation_id,created_at,attributes

``points`` and ``frames`` are JSON arrays with one entry per keyframe of a
track, and they are always the same length. ``frames`` holds *seconds*, not
frame indices, and may contain ``null`` (with a matching empty ``[]`` in
``points``) to mark a gap in the track. ``attributes`` is a JSON object with
the source video's ``width``/``height``/``size``/``mimetype``.

Geometry per keyframe:
    Rectangle  8 numbers, four possibly-rotated corners [x1,y1,x2,y2,x3,y3,x4,y4]
    Circle     3 numbers [cx, cy, radius]
    Point      2 numbers -- dropped, no extent
    LineString 2n numbers -- dropped, no width

Unlike ``old_scripts/video_stuff.py``, this module dispatches on ``shape_name``
rather than guessing from the coordinate-count, uses ``json.loads`` (which
handles the ``null`` entries that ``ast.literal_eval`` chokes on), and keeps
every keyframe paired with its own timestamp instead of collapsing a whole
track onto ``frames[0]``.
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

RECTANGLE = "Rectangle"
CIRCLE = "Circle"
SUPPORTED_SHAPES = (RECTANGLE, CIRCLE)

UMLAUT_MAP = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
}

# BIIGLE reports can carry very long `points` arrays for tracked annotations.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


@dataclass
class BoxAnnotation:
    """One axis-aligned box at one timestamp, in source-video pixel coordinates."""

    video_id: int
    video_filename: str
    meta_width: int
    meta_height: int
    label_name: str
    timestamp: float
    x1: float
    y1: float
    x2: float
    y2: float
    shape_name: str
    annotation_id: int
    keyframe: int
    source_csv: str


def iter_report_files(reports_dir):
    """Return the per-volume report CSVs, sorted.

    The glob deliberately requires a project subdirectory so the top-level
    ``counts.csv`` label histogram is not picked up as a report.
    """
    return sorted(glob.glob(os.path.join(reports_dir, "*", "*.csv")))


def _reject(rejects, row, source, reason, detail="", keyframe=""):
    rejects.append({
        "source_csv": source,
        "video_annotation_label_id": row.get("video_annotation_label_id", ""),
        "annotation_id": row.get("annotation_id", ""),
        "video_id": row.get("video_id", ""),
        "video_filename": row.get("video_filename", ""),
        "label_name": row.get("label_name", ""),
        "shape_name": row.get("shape_name", ""),
        "keyframe": keyframe,
        "reason": reason,
        "detail": detail,
    })


def parse_reports(reports_dir):
    """Parse every report CSV into box annotations.

    Args:
        reports_dir: Directory holding ``<project_id>_csv_video_annotation_report/``
            subdirectories.

    Returns:
        Tuple ``(annotations, rejects, row_count)`` where ``annotations`` is a
        list of :class:`BoxAnnotation`, ``rejects`` a list of dicts describing
        every dropped row/keyframe, and ``row_count`` the number of CSV rows read.
    """
    annotations = []
    rejects = []
    row_count = 0

    for path in iter_report_files(reports_dir):
        source = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                shape = (row.get("shape_name") or "").strip()
                if shape not in SUPPORTED_SHAPES:
                    _reject(rejects, row, source, "unsupported_shape", shape)
                    continue

                try:
                    points = json.loads(row["points"])
                    frames = json.loads(row["frames"])
                except (ValueError, KeyError) as error:
                    _reject(rejects, row, source, "unparsable_geometry", str(error))
                    continue

                if not isinstance(points, list) or not isinstance(frames, list):
                    _reject(rejects, row, source, "unparsable_geometry", "points/frames not lists")
                    continue
                if len(points) != len(frames):
                    _reject(rejects, row, source, "keyframe_length_mismatch",
                            "{} points vs {} frames".format(len(points), len(frames)))
                    continue

                try:
                    attributes = json.loads(row["attributes"])
                    meta_width = int(attributes["width"])
                    meta_height = int(attributes["height"])
                except (ValueError, KeyError, TypeError):
                    meta_width = meta_height = 0

                for index, (point, timestamp) in enumerate(zip(points, frames)):
                    if timestamp is None or not point:
                        _reject(rejects, row, source, "track_gap_keyframe", "", index)
                        continue

                    box = shape_to_box(shape, point)
                    if box is None:
                        _reject(rejects, row, source, "bad_geometry",
                                "{} with {} values".format(shape, len(point)), index)
                        continue

                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        _reject(rejects, row, source, "degenerate_box", str(box), index)
                        continue

                    annotations.append(BoxAnnotation(
                        video_id=int(row["video_id"]),
                        video_filename=row["video_filename"],
                        meta_width=meta_width,
                        meta_height=meta_height,
                        label_name=row["label_name"],
                        timestamp=float(timestamp),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        shape_name=shape,
                        annotation_id=int(row["annotation_id"]),
                        keyframe=index,
                        source_csv=source,
                    ))

    return annotations, rejects, row_count


def shape_to_box(shape, point):
    """Convert one keyframe's coordinate list to an axis-aligned box.

    Args:
        shape: ``"Rectangle"`` or ``"Circle"``.
        point: Flat list of coordinates for a single keyframe.

    Returns:
        ``(x1, y1, x2, y2)``, or ``None`` if the coordinate count is wrong or
        the circle radius is non-positive. 85% of the rectangles are rotated
        quads, so the min/max hull over the four corners is taken.
    """
    if shape == RECTANGLE:
        if len(point) != 8:
            return None
        xs = [float(v) for v in point[0::2]]
        ys = [float(v) for v in point[1::2]]
        return min(xs), min(ys), max(xs), max(ys)

    if shape == CIRCLE:
        if len(point) != 3:
            return None
        cx, cy, radius = (float(v) for v in point)
        if radius <= 0:
            return None
        return cx - radius, cy - radius, cx + radius, cy + radius

    return None


def sanitize_dir_name(name):
    """Turn a BIIGLE label name into a filesystem-safe directory name.

    Umlauts are transliterated rather than stripped so ``Stachelhäuter`` stays
    readable as ``Stachelhaeuter``.
    """
    for source, target in UMLAUT_MAP.items():
        name = name.replace(source, target)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name or "unnamed"


def build_class_index(annotations):
    """Assign stable global class IDs, most frequent label first.

    Args:
        annotations: List of :class:`BoxAnnotation`.

    Returns:
        List of dicts with ``class_id``, ``label_name``, ``dir_name`` and
        ``count``, ordered by descending count then label name. Directory-name
        collisions after sanitizing are broken with a numeric suffix.
    """
    counts = Counter(annotation.label_name for annotation in annotations)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    used = {}
    classes = []
    for class_id, (label_name, count) in enumerate(ordered):
        dir_name = sanitize_dir_name(label_name)
        if dir_name in used:
            used[dir_name] += 1
            dir_name = "{}_{}".format(dir_name, used[dir_name])
        else:
            used[dir_name] = 1
        classes.append({
            "class_id": class_id,
            "label_name": label_name,
            "dir_name": dir_name,
            "count": count,
        })
    return classes


def video_stem(video_filename):
    """Strip the extension from a BIIGLE video filename."""
    return os.path.splitext(video_filename)[0]
