"""Claude Sonnet 5 as a third zero-shot detector on the Thünen test split.

The Thünen zero-shot experiment pits NAUTILUS (underwater LMM) against Megalodon
(MBARI YOLOv8x, class-agnostic) on the North-Sea seafloor test split. Both do
badly. The thesis question is whether a *general* frontier model does any better
on this underwater domain -- i.e. is a domain-specific underwater foundation
model actually necessary?

This script adds Claude Sonnet 5 as a third detector, driven through headless
Claude Code (``claude -p``, subscription auth). It calls ``claude`` once per
image, hands the model the absolute image path (ingested through Claude's own
Read tool), and asks for a structured list of boxes via ``--json-schema``. The
task prompt is NAUTILUS's ``PROMPT_TEMPLATE`` verbatim, so the wording matches
the other two detectors; the operational rules (coordinate convention, no
commentary) go in ``--append-system-prompt`` to keep the task prompt identical.

Output is the canonical ``batch_inference`` layout -- ``results/<stem>.txt``
(JSON-lines, one box per line, original-pixel coords) plus a sibling
``metadata.json`` -- so ``evaluate_detections.py`` scores it unchanged, in the
same prompt label space as NAUTILUS and Megalodon. Boxes are written in
original-pixel space with ``input_* == original_*`` so the evaluator's rescale is
x1 (the ``yolo_scaling.write_nautilus_format`` convention).

Context isolation: each ``claude`` subprocess runs from the run-dir as cwd (out
of the repo, so no ``CLAUDE.md`` / ``AGENTS.md`` auto-discovery), with
``--safe-mode`` (no hooks, no customizations, keeps the OAuth subscription).
``--bare`` is *not* usable -- it forces ``ANTHROPIC_API_KEY`` auth only.

Non-determinism: Sonnet 5 rejects ``temperature``; re-runs differ. That is
expected, not a bug -- note it when reporting the number.

Resumable, and quota-aware. ``--skip-existing`` (default on) skips any stem that
already has a ``results/<stem>.txt``, so a killed run resumes by re-running the
exact same command. If the subscription usage limit is hit mid-run the script
stops cleanly (exit code 2) rather than burning retries on every remaining
image -- the un-run stems are simply left for the next resume. A full-split run
(~1300 images) will not finish inside one 5-hour usage window; just re-run it
each window until ``nothing to do``.

Usage:
    # host-side, torch-free -- runs on beta directly (no container)
    cd /home/tfricke/nautilus/NAUTILUS/qwen-vl-finetune/scripts

    # 1. smoke test: 3 images, serial, throwaway run-dir
    python3 thuenen_pipeline/claude_zeroshot.py --limit 3 --concurrency 1 \
      --run-dir /tmp/claude_smoke

    # 2. 250-image pilot on the pinned screening subsample
    python3 thuenen_pipeline/claude_zeroshot.py \
      --run-dir /home/tfricke/nautilus/runs/thuenen_claude_sonnet5 --concurrency 3

    # 2b. full test split into the same run-dir -- incremental via --skip-existing.
    #     Exits 2 when the usage limit is hit; just re-run until "nothing to do".
    python3 thuenen_pipeline/claude_zeroshot.py --full \
      --run-dir /home/tfricke/nautilus/runs/thuenen_claude_sonnet5 --concurrency 3

    # 3. score it (in the container -- evaluate_detections.py needs torch-free PIL only,
    #    but the house convention is to run it there)
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \
      --pred-dir     /workspace/runs/thuenen_claude_sonnet5/results \
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \
      --save-json    /workspace/runs/thuenen_claude_sonnet5/metrics.json
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

from PIL import Image

# Sibling imports. nautilus_zeroshot is torch-free at module level (it imports
# torch only inside run_zero_shot), so this is safe on the host. batch_inference
# is *not* torch-free, so find_images is reimplemented below rather than imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nautilus_zeroshot import PROMPT_TEMPLATE, snap_labels  # noqa: E402

DEFAULT_DATASET = "/home/tfricke/nautilus/datasets/thuenen_scaling"
DEFAULT_RUN_DIR = "/home/tfricke/nautilus/runs/thuenen_claude_sonnet5"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Operational rules only -- kept out of the task prompt so the task wording stays
# byte-identical to what NAUTILUS and Megalodon are given.
OPERATIONAL_RULES = (
    "Read the image file at the path given in the user message. Return only the "
    "structured object. Each detection's `bbox_2d` is [x1, y1, x2, y2] as "
    "fractions of image width and height in [0, 1], top-left corner first then "
    "bottom-right corner. Emit one entry per detected object; return an empty "
    "list if there are none. Do not use any tool other than Read. No commentary."
)

# Enforced by --json-schema: the result comes back parsed in `structured_output`.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string"},
                },
                "required": ["bbox_2d", "label"],
            },
        }
    },
    "required": ["detections"],
}


def find_images(image_dir, recursive=False):
    """Non-recursive image discovery, matching batch_inference.find_images."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in image_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


# ── the per-image claude call ────────────────────────────────────────────────
class ClaudeError(RuntimeError):
    """A claude invocation that failed in a way worth retrying."""


class QuotaExhausted(RuntimeError):
    """The subscription usage limit was hit -- not retryable, and no point
    starting any further images. The driver stops the run cleanly and the
    remaining stems are picked up on the next --skip-existing resume."""


# Substrings that mean "the subscription is spent for now" rather than a
# transient hiccup. A plain HTTP 429 / "rate limit" is *not* here on purpose --
# that one is short-lived and worth retrying.
_QUOTA_MARKERS = (
    "usage limit reached",
    "exceeded your usage",
    "reached your usage limit",
    "out of credits",
    "insufficient credit",
    "upgrade to increase your usage limit",
)


def _looks_like_quota(*texts):
    blob = " ".join(t for t in texts if t).lower()
    return any(marker in blob for marker in _QUOTA_MARKERS)


def build_command(task_prompt, image_path, model, max_budget_usd):
    # The task prompt is the NAUTILUS wording verbatim; the absolute image path
    # is appended per-image on its own line so Claude's Read tool can load it.
    prompt = "{}\n\nImage: {}".format(task_prompt, image_path)
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--json-schema", json.dumps(OUTPUT_SCHEMA),
        "--append-system-prompt", OPERATIONAL_RULES,
        "--tools", "Read",
        "--permission-mode", "dontAsk",
        "--safe-mode",
        "--no-session-persistence",
        "--add-dir", str(image_path.parent),
    ]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    return cmd


def invoke_claude(cmd, cwd, timeout):
    """Run one claude subprocess, return the parsed JSON envelope."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ClaudeError("timed out after {}s".format(timeout)) from error

    if proc.returncode != 0:
        if _looks_like_quota(proc.stderr, proc.stdout):
            raise QuotaExhausted((proc.stderr or proc.stdout or "").strip()[:300])
        raise ClaudeError(
            "exit {}: {}".format(proc.returncode, (proc.stderr or proc.stdout or "").strip()[:500])
        )
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        if _looks_like_quota(proc.stdout, proc.stderr):
            raise QuotaExhausted(proc.stdout.strip()[:300]) from error
        raise ClaudeError("stdout is not JSON: {}".format(proc.stdout.strip()[:500])) from error

    if _looks_like_quota(str(envelope.get("result")), envelope.get("subtype")):
        raise QuotaExhausted(str(envelope.get("result"))[:300])

    if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
        raise ClaudeError(
            "is_error={} subtype={} result={}".format(
                envelope.get("is_error"), envelope.get("subtype"),
                str(envelope.get("result"))[:300],
            )
        )
    return envelope


def detections_from_envelope(envelope):
    """Pull the detections list out of the structured output."""
    obj = envelope.get("structured_output")
    if obj is None:
        # --json-schema should always populate structured_output on success; fall
        # back to parsing `result` as a last resort.
        try:
            obj = json.loads(envelope["result"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ClaudeError("no structured_output and result is not JSON") from error
    dets = obj.get("detections", [])
    if not isinstance(dets, list):
        raise ClaudeError("`detections` is not a list: {!r}".format(dets))
    return dets


def call_with_retries(cmd, cwd, timeout, max_retries):
    """invoke_claude + exponential backoff. Returns (envelope, n_attempts)."""
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            envelope = invoke_claude(cmd, cwd, timeout)
            detections_from_envelope(envelope)  # validate before we commit
            return envelope, attempt
        except ClaudeError as error:
            last = error
            if attempt < max_retries:
                time.sleep(min(60, 2 ** attempt))
    raise last


# ── box post-processing (mirrors yolo_scaling.write_nautilus_format) ─────────
def boxes_to_lines(detections, orig_w, orig_h):
    """Fraction boxes -> original-pixel JSON-lines body (no enclosing list)."""
    lines = []
    for det in detections:
        bbox = det.get("bbox_2d")
        label = det.get("label")
        if not isinstance(bbox, list) or len(bbox) != 4 or label is None:
            continue
        try:
            fx1, fy1, fx2, fy2 = (min(1.0, max(0.0, float(v))) for v in bbox)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((round(fx1 * orig_w), round(fx2 * orig_w)))
        y1, y2 = sorted((round(fy1 * orig_h), round(fy2 * orig_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(json.dumps(
            {"bbox_2d": [x1, y1, x2, y2], "label": str(label)}, ensure_ascii=False))
    return "\n".join(lines)


def process_image(stem, image_path, run_dir, prompt_names, task_prompt, args,
                  stop_event=None):
    """Full pipeline for one image. Returns a result dict (never raises)."""
    result = {"stem": stem, "ok": False, "attempts": 0, "unmatched": 0,
              "cost_usd": 0.0, "usage": {}, "error": None, "quota": False,
              "skipped": False, "dims": None, "name": image_path.name}
    if stop_event is not None and stop_event.is_set():
        result["skipped"] = True
        return result
    cmd = build_command(task_prompt, image_path, args.model, args.max_budget_usd)
    try:
        envelope, attempts = call_with_retries(
            cmd, run_dir, args.timeout, args.max_retries)
        result["attempts"] = attempts
    except QuotaExhausted as error:
        result["quota"] = True
        result["error"] = "usage limit reached: {}".format(error)
        return result
    except ClaudeError as error:
        result["error"] = str(error)
        return result

    result["cost_usd"] = float(envelope.get("total_cost_usd") or 0.0)
    result["usage"] = envelope.get("usage") or {}

    detections = detections_from_envelope(envelope)
    with Image.open(image_path) as image:
        orig_w, orig_h = image.size

    body = boxes_to_lines(detections, orig_w, orig_h)
    snapped, n_unmatched = snap_labels(body, prompt_names)
    result["unmatched"] = n_unmatched

    results_dir = os.path.join(run_dir, "results")
    raw_dir = os.path.join(run_dir, "results_raw")
    with open(os.path.join(raw_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
        handle.write(body)
    with open(os.path.join(results_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
        handle.write(snapped)

    result["dims"] = {
        "input_height": orig_h, "input_width": orig_w,
        "original_width": orig_w, "original_height": orig_h,
    }
    result["ok"] = True
    return result


# ── driver ───────────────────────────────────────────────────────────────────
def resolve_stems(args):
    """Return the ordered list of stems to run, and the subset sha1 (or None)."""
    split_images = os.path.join(args.dataset, args.split, "images")
    all_paths = find_images(__import__("pathlib").Path(split_images))
    by_stem = {p.stem: p for p in all_paths}
    if not by_stem:
        raise SystemExit("no images under {}".format(split_images))

    if args.full:
        stems = sorted(by_stem)
        sha1 = None
    else:
        subset_path = args.subset or os.path.join(args.dataset, "screening_subsample.json")
        with open(subset_path, encoding="utf-8") as handle:
            subset = json.load(handle)
        stems = list(subset["stems"])
        sha1 = subset.get("sha1")
        missing = [s for s in stems if s not in by_stem]
        if missing:
            raise SystemExit(
                "{} subset stems are not in {}: {}".format(
                    len(missing), split_images, ", ".join(missing[:5])))

    if args.limit:
        stems = stems[:args.limit]
    return [(s, by_stem[s]) for s in stems], sha1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--subset", default=None,
                        help="JSON with a ['stems'] list; default "
                             "<dataset>/screening_subsample.json.")
    parser.add_argument("--full", action="store_true",
                        help="Run every image in <split>/images instead of the subset.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N stems (smoke test).")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false",
                        help="Re-run stems that already have a results/ file.")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-image claude subprocess timeout in seconds.")
    parser.add_argument("--max-budget-usd", type=float, default=None,
                        help="Passthrough to `claude --max-budget-usd`.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    results_dir = os.path.join(run_dir, "results")
    raw_dir = os.path.join(run_dir, "results_raw")
    for directory in (results_dir, raw_dir):
        os.makedirs(directory, exist_ok=True)
    metadata_path = os.path.join(run_dir, "metadata.json")

    prompt_names = read_lines(os.path.join(args.dataset, "classes_prompt.txt"))
    task_prompt = PROMPT_TEMPLATE.format(classes=", ".join(prompt_names))
    print("{} prompt classes".format(len(prompt_names)))
    print("task prompt: {}".format(task_prompt))

    stems, subset_sha1 = resolve_stems(args)

    # Seed / reload metadata so a killed run resumes.
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.setdefault("image_dims", {})
    else:
        metadata = {"prompt": task_prompt, "checkpoint": "claude-sonnet-5",
                    "image_dims": {}}
    metadata["prompt"] = task_prompt
    metadata["checkpoint"] = "claude-sonnet-5"
    metadata["model"] = args.model
    metadata["subset_sha1"] = subset_sha1
    metadata["append_system_prompt"] = OPERATIONAL_RULES
    try:
        metadata["cli_version"] = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        metadata["cli_version"] = None

    # Self-heal: a run killed uncleanly (SIGKILL/SIGINT) can leave a
    # results/<stem>.txt on disk whose image_dims entry was never flushed to
    # metadata.json. --skip-existing would then skip that stem forever and
    # evaluate_detections.py silently zeroes any evaluated image missing from
    # image_dims. Backfill those entries from the image files themselves
    # (boxes are original-pixel with input == original, so this is exact).
    healed = 0
    for stem, path in stems:
        if (os.path.exists(os.path.join(results_dir, stem + ".txt"))
                and path.name not in metadata["image_dims"]):
            with Image.open(path) as image:
                width, height = image.size
            metadata["image_dims"][path.name] = {
                "input_height": height, "input_width": width,
                "original_width": width, "original_height": height,
            }
            healed += 1
    if healed:
        print("backfilled {} missing image_dims entries".format(healed))

    todo = stems
    if args.skip_existing:
        todo = [(s, p) for s, p in stems
                if not os.path.exists(os.path.join(results_dir, s + ".txt"))]
    print("{} stems total, {} to run ({} already done)".format(
        len(stems), len(todo), len(stems) - len(todo)))
    if not todo:
        if healed:
            tmp = metadata_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)
            os.replace(tmp, metadata_path)
        print("nothing to do")
        return 0

    stats = Counter()
    # Seed the running totals from any prior resume so they stay cumulative.
    usage_totals = Counter(metadata.get("usage_totals") or {})
    cost_total = [float(metadata.get("cost_usd_total") or 0.0)]
    failures = list(metadata.get("failures", []))
    prior_unmatched = int(metadata.get("n_unmatched_total") or 0)
    lock = threading.Lock()
    done = [0]
    stop_event = threading.Event()  # set when the subscription quota runs out

    def flush():
        metadata["n_unmatched_total"] = prior_unmatched + stats["unmatched"]
        metadata["usage_totals"] = dict(usage_totals)
        metadata["cost_usd_total"] = round(cost_total[0], 4)
        metadata["failures"] = sorted(set(failures))
        tmp = metadata_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, metadata_path)

    def handle_result(res):
        with lock:
            if res.get("skipped"):
                stats["skipped"] += 1
                return
            done[0] += 1
            if res["ok"]:
                stats["ok"] += 1
                stats["unmatched"] += res["unmatched"]
                metadata["image_dims"][res["name"]] = res["dims"]
                failures[:] = [f for f in failures if f != res["stem"]]
            elif res.get("quota"):
                # Not a failure -- just not done. Leave it out of `failures`
                # so the next --skip-existing resume retries it cleanly.
                stats["quota"] += 1
                if not stop_event.is_set():
                    print("\n[quota] subscription usage limit reached -- "
                          "stopping. Re-run the same command later to resume "
                          "(--skip-existing picks up where this left off).")
                    print("[quota] {}".format(res["error"]))
                stop_event.set()
            else:
                stats["failed"] += 1
                failures.append(res["stem"])
                print("[failed] {}: {}".format(res["stem"], res["error"]))
            cost_total[0] += res["cost_usd"]
            for key in ("input_tokens", "output_tokens",
                        "cache_read_input_tokens", "cache_creation_input_tokens"):
                if isinstance(res["usage"].get(key), int):
                    usage_totals[key] += res["usage"][key]
            if done[0] % 25 == 0:
                flush()
                print("  {}/{} done, ok={} failed={} ${:.2f}".format(
                    done[0], len(todo), stats["ok"], stats["failed"], cost_total[0]))

    flush()  # persist any backfilled image_dims before the first checkpoint

    start = time.time()
    if args.concurrency <= 1:
        for stem, path in todo:
            if stop_event.is_set():
                break
            handle_result(process_image(
                stem, path, run_dir, prompt_names, task_prompt, args, stop_event))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(process_image, stem, path, run_dir,
                            prompt_names, task_prompt, args, stop_event)
                for stem, path in todo
            ]
            for future in as_completed(futures):
                try:
                    res = future.result()
                except CancelledError:
                    continue
                handle_result(res)
                if stop_event.is_set():
                    for pending in futures:
                        pending.cancel()

    flush()
    elapsed = time.time() - start
    remaining = len(todo) - stats["ok"] - stats["failed"]
    print("\ndone in {:.0f}s -- ok={} failed={} remaining={}, {} unmatched labels, "
          "${:.2f}".format(elapsed, stats["ok"], stats["failed"], remaining,
                           stats["unmatched"], cost_total[0]))
    print("results:  {}".format(results_dir))
    print("metadata: {}".format(metadata_path))
    if failures:
        print("{} failures recorded in metadata.json -- re-run to retry them".format(
            len(set(failures))))
    if stop_event.is_set():
        print("STOPPED EARLY on usage limit -- {} stems still to do. Re-run the "
              "exact same command to resume.".format(remaining))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
