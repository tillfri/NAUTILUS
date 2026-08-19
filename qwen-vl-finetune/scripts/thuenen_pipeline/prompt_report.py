"""Compare the Thünen prompt/few-shot screening runs on adherence, not on mAP.

The zero-shot baseline scores class-aware mAP@0.5 = 0.0007. Nothing can be steered
on a number that small, and a variant that halves or doubles it has told you nothing.
What *is* measurable is whether the instruction reaches the model at all:

* **in-vocabulary share** -- what fraction of predicted labels are one of the 24
  prompt classes. Computed on ``results_raw/``, never on ``results/``:
  ``snap_labels`` canonicalises case and then falls back to substring matching, so it
  would quietly fold "sea star" onto "sand star" and "organism" onto "unidentified
  organism". The snapped text can only ever bound this metric from above.
* **distinct classes used**, out of 24 -- the baseline emitted 6.
* **top-1 label share** -- the "everything is a starfish" collapse as one number.
* **boxes per image** and the **empty-output rate** -- whether an abstention
  instruction produces any abstention at all.

Localization is reported second, from ``localization_report.py``, and mAP third from
``evaluate_detections.py``: both are context for the adherence numbers, not the
signal. Prediction parsing goes through ``visualize_detections.extract_detections_from_text``
so it cannot drift from what the scorers see.

``--diff A B`` is a separate mode: the correctness gate for the runner in
``prompt_experiments.py``. It compares a fresh ``P0_baseline`` against
``P0_baseline_recovered`` (the subsample carved out of the finished full-split run).
Decoding is effectively greedy -- ``generation_config.json`` sets ``do_sample: true``
with ``top_k: 1`` -- so the two should agree closely; the residual is GPU float
non-associativity, and the runs were made on different GPU generations (the baseline
on beta's Ada 4090s, the sweep on alpha's Ampere 3090s), so expect a small non-zero
drift rather than byte-equality. A *large* diff means the two-pass rescaling or the
image-token doubling is wrong.

Usage:
    # the comparison table over every run dir that has results
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_report.py \
      --save-json /workspace/runs/thuenen_prompt/comparison.json \
      --markdown  /workspace/runs/thuenen_prompt/comparison.md

    # adherence only, skipping the mAP/localization subprocesses
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_report.py --no-metrics

    # the runner correctness gate
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
thuenen_pipeline/prompt_report.py \
      --diff /workspace/runs/thuenen_prompt/P0_baseline \
             /workspace/runs/thuenen_prompt/P0_baseline_recovered
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

from evaluate_detections import greedy_match  # noqa: E402
from nautilus_zeroshot import canonicalise  # noqa: E402
from prompt_experiments import DEFAULT_ROOT, DEFAULT_RUNS, read_prompt_classes  # noqa: E402
from visualize_detections import extract_detections_from_text, rescale_bbox  # noqa: E402

try:
    from prompt_variants import SWEEP
except ImportError:  # pragma: no cover
    SWEEP = []


# ── adherence ────────────────────────────────────────────────────────────────
def read_predictions(run_dir, prefer_raw=True):
    """Return ``({stem: text}, source_dir_name)``.

    ``results_raw/`` is the honest source. ``P0_baseline_recovered`` has none -- it
    is carved out of a run that only ever wrote snapped text -- so it falls back to
    ``results/`` and the caller marks the row as an upper bound.
    """
    for name in (("results_raw", "results") if prefer_raw else ("results",)):
        directory = os.path.join(run_dir, name)
        if os.path.isdir(directory):
            files = [f for f in sorted(os.listdir(directory)) if f.endswith(".txt")]
            if files:
                return ({os.path.splitext(f)[0]:
                         open(os.path.join(directory, f), encoding="utf-8").read()
                         for f in files}, name)
    return {}, None


# A coordinate quadruple in *any* bracket style, and the bare word in front of it.
# Needed because several variants stop emitting JSON entirely and fall back to
# "fish [957, 461, 1098, 510]" / "starfish (579, 628, 696, 695)" / "starfish = [...]".
# extract_detections_from_text sees none of that, so counting only strict matches
# would score a format change as an abstention -- which is a different finding.
ANY_BOX = re.compile(r"[\[(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\])]")
ANY_BOX_GROUPS = re.compile(
    r"[\[(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\])]")
BARE_LABELLED_BOX = re.compile(
    r"([A-Za-z][A-Za-z0-9 _-]{1,30}?)\s*[=:]?\s*[\[(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\])]")
NO_DETECTION_TEXTS = {"", "[]", "[ ]", "none", "none.", "no objects", "no objects."}


def parse_any(text):
    """``(labels, n_boxes_any, n_boxes_json)`` for one response.

    The strict JSON parse is authoritative for labels when it finds anything -- it is
    the same function the scorers use. Only when it comes back empty do we fall back
    to the bare-coordinate form, so a variant that merely reformats its output is
    still measured on vocabulary rather than silently counted as having abstained.
    """
    detections = extract_detections_from_text(text)
    n_json = len(detections)
    n_any = max(len(ANY_BOX.findall(text)), n_json)
    if detections:
        return [str(d.get("label", "")) for d in detections], n_any, n_json
    return [m.strip() for m in BARE_LABELLED_BOX.findall(text)], n_any, n_json


def adherence(texts, prompt_names):
    """The primary metrics, over raw model text.

    Two in-vocabulary numbers are reported. ``exact`` requires the label verbatim;
    ``canonical`` accepts it through ``nautilus_zeroshot.canonicalise`` (lowercase,
    punctuation collapsed), because "Brittle star" is the right class named in the
    wrong case -- a formatting miss, not a vocabulary miss. ``canonical`` is the
    headline. Neither uses substring matching, which is what makes this recoverable
    only from the unsnapped text.
    """
    canonical = {canonicalise(name): name for name in prompt_names}
    exact_names = set(prompt_names)

    labels, n_boxes, n_json, empty, explicit_empty = [], 0, 0, 0, 0
    for text in texts.values():
        found, boxes_any, boxes_json = parse_any(text)
        n_boxes += boxes_any
        n_json += boxes_json
        if boxes_any == 0:
            empty += 1
            if text.strip().lower() in NO_DETECTION_TEXTS:
                explicit_empty += 1
        labels.extend(found)

    total = len(labels) or 1
    in_exact = sum(1 for l in labels if l in exact_names)
    hits = [canonical[canonicalise(l)] for l in labels if canonicalise(l) in canonical]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0] if counts else ("", 0)
    out_of_vocab = Counter(l for l in labels if canonicalise(l) not in canonical)

    return {
        "images": len(texts),
        "boxes": n_boxes,
        "boxes_json": n_json,
        # What fraction of the boxes arrived in the JSON shape the scorers can read.
        # Below 1.0 the variant changed format, and evaluate_detections.py will score
        # the missing ones as if the model had found nothing.
        "json_format_frac": round(n_json / max(n_boxes, 1), 4),
        "boxes_per_image": round(n_boxes / max(len(texts), 1), 3),
        "empty_outputs": empty,
        "empty_output_frac": round(empty / max(len(texts), 1), 4),
        "explicit_empty_outputs": explicit_empty,
        "labels": len(labels),
        "in_vocab_exact": in_exact,
        "in_vocab_exact_frac": round(in_exact / total, 4),
        "in_vocab_canonical": len(hits),
        "in_vocab_canonical_frac": round(len(hits) / total, 4),
        "distinct_classes_used": len(set(hits)),
        "distinct_classes_total": len(prompt_names),
        "top1_label": top_label,
        "top1_frac": round(top_count / total, 4),
        "out_of_vocab": dict(out_of_vocab.most_common(10)),
        "label_histogram": dict(counts.most_common()),
    }


# ── few-shot: is it learning, or just replaying the demonstration? ───────────
EXEMPLAR_INPUT_W, EXEMPLAR_INPUT_H = 588, 308  # 2704x1520 at the 256-patch budget


def exemplar_replay(run_dir, root, exemplar_stems, texts):
    """Fraction of answers that are a demonstration played back verbatim.

    The k=2 run returns the *first* exemplar's box set character for character, in
    that exemplar's coordinate space rather than the query's -- so the model is
    reproducing the prompt, not reading the image. An adherence score computed over
    replayed text measures the demonstration, so the table has to show how much of
    each F* row is replay before its in-vocabulary number means anything.

    Returns ``None`` for prompt-only variants (nothing to replay).
    """
    if not exemplar_stems:
        return None
    path = os.path.join(root, "exemplars_k{}.json".format(len(exemplar_stems)))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        exemplars = json.load(handle)

    def shown(exemplar):
        w = EXEMPLAR_INPUT_W / exemplar["original_width"]
        h = EXEMPLAR_INPUT_H / exemplar["original_height"]
        return {(round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h))
                for box in exemplar["boxes"]
                for x1, y1, x2, y2 in [box["bbox_2d"]]}

    demonstrated = [shown(e) for e in exemplars]
    every_box = set().union(*demonstrated) if demonstrated else set()

    n = exact = reused = 0
    for text in texts.values():
        boxes = {tuple(int(v) for v in m) for m in ANY_BOX_GROUPS.findall(text)}
        if not boxes:
            continue
        n += 1
        exact += boxes in demonstrated
        reused += bool(boxes & every_box)
    if not n:
        return None
    return {"images_with_boxes": n,
            "replayed_a_demonstration_exactly": exact,
            "replay_frac": round(exact / n, 4),
            "reused_a_demonstrated_box": reused,
            "reused_frac": round(reused / n, 4)}


# ── secondary metrics, via the existing scorers ──────────────────────────────
def run_scorer(script, out_json, args, extra):
    """Run one of the existing scorers and read back its --save-json.

    Called as a subprocess rather than imported so the numbers in this table come
    from exactly the code path the pipeline uses everywhere else.
    """
    if os.path.isfile(out_json) and not args.rescore:
        with open(out_json, encoding="utf-8") as handle:
            return json.load(handle)
    command = [sys.executable, os.path.join(SCRIPTS, script),
               "--gt-dir", os.path.join(args.root, args.split, "labels_prompt"),
               "--image-dir", os.path.join(args.root, args.split, "images"),
               "--classes-file", os.path.join(args.root, "classes_prompt.txt"),
               "--save-json", out_json] + extra
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not os.path.isfile(out_json):
        print("[warning] {} failed for {}".format(script, extra), file=sys.stderr)
        return None
    with open(out_json, encoding="utf-8") as handle:
        return json.load(handle)


def secondary(run_dir, args):
    """mAP from evaluate_detections.py, recall from localization_report.py."""
    results = os.path.join(run_dir, "results")
    metrics = run_scorer("evaluate_detections.py", os.path.join(run_dir, "metrics.json"),
                         args, ["--pred-dir", results])
    localization = run_scorer("localization_report.py",
                              os.path.join(run_dir, "localization.json"),
                              args, ["--pred-dir", results])
    out = {}
    if metrics:
        out["mAP50_agnostic"] = metrics["class_agnostic"]["mAP@0.5"]
        out["mAP50_aware"] = metrics["class_aware"]["mAP@0.5"]
    if localization:
        out["recall_iou50"] = localization["criteria"]["iou_0.5"]["recall"]
        out["recall_centre"] = localization["criteria"]["centre_in_box"]["recall"]
    return out


# ── the report ───────────────────────────────────────────────────────────────
def discover(args):
    """Run dirs to report on, in SWEEP order with anything else appended."""
    if args.variants:
        names = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        names = [d for d in sorted(os.listdir(args.runs))
                 if os.path.isdir(os.path.join(args.runs, d))
                 and not d.startswith("_")
                 and os.path.isdir(os.path.join(args.runs, d, "results"))]
    order = {name: i for i, name in enumerate(SWEEP)}
    return sorted(names, key=lambda n: (order.get(n, len(SWEEP)), n))


MARKDOWN_HEADER = (
    "| variant | box/img | JSON | empty | in-vocab | classes | top-1 label | top-1 | "
    "replay | R@.5 | R@ctr | mAP50 agn | mAP50 aware | tokens |")
MARKDOWN_RULE = "|" + "---|" * 14


def markdown_row(name, row):
    adh, sec = row["adherence"], row.get("secondary", {})
    tokens = row.get("prompt_tokens", {})
    def pct(value):
        return "-" if value is None else "{:.1f}%".format(100 * value)
    def num(value, digits=4):
        return "-" if value is None else "{:.{d}f}".format(value, d=digits)
    star = "*" if row.get("upper_bound") else ""
    return ("| {} | {:.2f} | {} | {} | {}{} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} |"
            ).format(
        name, adh["boxes_per_image"], pct(adh["json_format_frac"]),
        pct(adh["empty_output_frac"]),
        pct(adh["in_vocab_canonical_frac"]), star,
        adh["distinct_classes_used"], adh["distinct_classes_total"],
        adh["top1_label"] or "-", pct(adh["top1_frac"]),
        pct(row["exemplar_replay"]["replay_frac"]) if "exemplar_replay" in row else "-",
        num(sec.get("recall_iou50")), num(sec.get("recall_centre")),
        num(sec.get("mAP50_agnostic")), num(sec.get("mAP50_aware")),
        "{}/{}".format(tokens.get("mean", "-"), tokens.get("max", "-")))


def build_report(args):
    prompt_names = read_prompt_classes(args.root)
    report = {"root": args.root, "runs": args.runs, "variants": {}}

    for name in discover(args):
        run_dir = os.path.join(args.runs, name)
        texts, source = read_predictions(run_dir)
        if not texts:
            print("[skip] {}: no predictions".format(name), file=sys.stderr)
            continue
        row = {
            "run_dir": run_dir,
            "prediction_source": source,
            # results/ is snapped: substring matching already moved labels onto
            # prompt names, so its in-vocabulary share is an upper bound only.
            "upper_bound": source == "results",
            "adherence": adherence(texts, prompt_names),
        }
        metadata_path = os.path.join(run_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            tokens = list(metadata.get("prompt_tokens", {}).values())
            if tokens:
                row["prompt_tokens"] = {"mean": int(sum(tokens) / len(tokens)),
                                        "max": max(tokens)}
            for key in ("note", "subsample_sha1", "exemplars", "snapped"):
                if key in metadata:
                    row[key] = metadata[key]
        replay = exemplar_replay(run_dir, args.root, row.get("exemplars"), texts)
        if replay:
            row["exemplar_replay"] = replay
        if not args.no_metrics:
            row["secondary"] = secondary(run_dir, args)
        report["variants"][name] = row

    shas = {n: r.get("subsample_sha1") for n, r in report["variants"].items()
            if r.get("subsample_sha1")}
    if len(set(shas.values())) > 1:
        print("[warning] runs do not share one subsample -- the table compares "
              "different image sets: {}".format(shas), file=sys.stderr)

    lines = [MARKDOWN_HEADER, MARKDOWN_RULE]
    lines += [markdown_row(name, row) for name, row in report["variants"].items()]
    if any(r.get("upper_bound") for r in report["variants"].values()):
        lines.append("")
        lines.append("`*` in-vocabulary share read off snapped text -- an upper "
                     "bound, not the true share (no `results_raw/` for that run).")
    table = "\n".join(lines)
    print(table)

    for name, row in report["variants"].items():
        out_of_vocab = row["adherence"]["out_of_vocab"]
        if out_of_vocab:
            print("\n{} out-of-vocabulary labels: {}".format(
                name, ", ".join("{} x{}".format(k, v)
                                for k, v in out_of_vocab.items())), file=sys.stderr)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print("\nwrote {}".format(args.markdown), file=sys.stderr)
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("wrote {}".format(args.save_json), file=sys.stderr)
    return 0


# ── the runner correctness gate ──────────────────────────────────────────────
def to_original(run_dir, stem, detections):
    """Move boxes from model-input space back to original pixels.

    Both runs write model-space coordinates (the ``batch_inference`` layout), so a
    naive text diff would still be valid if the pixel budgets matched -- but they
    need not, and a silent budget change is exactly the kind of thing this gate
    exists to catch. Rescaling both sides makes the comparison independent of it.
    """
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as handle:
        dims = json.load(handle).get("image_dims", {})
    entry = dims.get(stem + ".jpg")
    if entry is None:
        return [d["bbox_2d"] for d in detections]
    inv_w = entry["original_width"] / entry["input_width"]
    inv_h = entry["original_height"] / entry["input_height"]
    return [rescale_bbox(d["bbox_2d"], inv_w, inv_h) for d in detections]


def diff_runs(args):
    left, right = args.diff
    left_texts, _ = read_predictions(left, prefer_raw=False)
    right_texts, _ = read_predictions(right, prefer_raw=False)
    common = sorted(set(left_texts) & set(right_texts))
    if not common:
        raise SystemExit("no stems in common between {} and {}".format(left, right))

    identical, boxes_identical = 0, 0
    added = removed = unchanged = 0
    label_changes = Counter()
    for stem in common:
        lt, rt = left_texts[stem], right_texts[stem]
        if lt == rt:
            identical += 1
        left_dets = extract_detections_from_text(lt)
        right_dets = extract_detections_from_text(rt)
        left_boxes = to_original(left, stem, left_dets)
        right_boxes = to_original(right, stem, right_dets)
        left_labels = [str(d.get("label", "")) for d in left_dets]
        right_labels = [str(d.get("label", "")) for d in right_dets]
        # class_aware=False: the gate asks whether the same boxes came back, and a
        # label that moved on an otherwise identical box is reported separately.
        matched, only_left, only_right = greedy_match(
            left_boxes, left_labels, right_boxes, right_labels,
            args.iou_threshold, False)
        unchanged += len(matched)
        added += len(only_left)
        removed += len(only_right)
        if not only_left and not only_right and left_boxes:
            boxes_identical += 1
        for left_idx, right_idx, _iou in matched:
            if left_labels[left_idx] != right_labels[right_idx]:
                label_changes["{} -> {}".format(
                    right_labels[right_idx], left_labels[left_idx])] += 1

    print("diff {} vs {}".format(os.path.basename(left.rstrip("/")),
                                 os.path.basename(right.rstrip("/"))))
    print("  {} stems in common".format(len(common)))
    print("  text identical      {:4d}  ({:.1f}%)".format(
        identical, 100.0 * identical / len(common)))
    print("  boxes fully matched {:4d}  ({:.1f}%) at IoU {}".format(
        boxes_identical, 100.0 * boxes_identical / len(common), args.iou_threshold))
    print("  boxes: {} matched, {} only in left, {} only in right".format(
        unchanged, added, removed))
    if label_changes:
        print("  label changes on matched boxes:")
        for change, count in label_changes.most_common(10):
            print("    {} x{}".format(change, count))
    print("\n  Decoding is effectively greedy (top_k=1), so a small drift is GPU "
          "float non-associativity;\n  a large one means the runner's rescaling or "
          "image-token doubling is wrong.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", default="",
                        help="Comma-separated run-dir names; default is every run "
                             "dir under --runs that has results.")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Adherence only; skip the mAP/localization scorers.")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-run the scorers even if their json already exists.")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--diff", nargs=2, metavar=("LEFT", "RIGHT"), default=None,
                        help="Compare two run dirs instead of building the table.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    return diff_runs(args) if args.diff else build_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
