"""Reproduce the NAUTILUS paper's NautData benchmarks, with the VFE ablatable.

The paper's Table 2 (8 tasks, NautData test), Table 3 (counting, IOCfish5k -- the
counting bucket of the same test JSON) and Table 7 (grounding PR@0.5 on the
degraded-condition subsets) are all produced by ``NAUTILUS/eval/eval4nautilus_qwen.py``.
This script is that evaluation, unchanged where it matters and extended where it
has to be:

* **Inference is the upstream loop, verbatim.** Same processor limits (min 1*28^2,
  max 1338*28^2), same ``process_vision_info`` path with no per-message
  ``max_pixels``, same tag-2/7 prompt-box rescale via upstream ``get_grid_thw``,
  same ``max_new_tokens=2048``, same prediction record. That includes the known
  ``get_grid_thw`` defect (original image preprocessed, not the 28-rounded one the
  model sees) -- it is left in on purpose, because the numbers being reproduced were
  made with it. ``metadata.json`` says so.
* **Scoring is the upstream ``Metric_method_dict``**, imported, not copied, with
  ``java=True`` so METEOR (the paper's caption metric) is computed. One input fix:
  newlines inside a caption/VQA answer become spaces first. pycocoevalcap's METEOR
  speaks a line protocol to its jar, so a mid-answer newline desyncs it, raises
  while it holds its lock, and its ``__del__`` then deadlocks the scorer forever.
  ``process_text`` already strips trailing ones, so this is a no-op for every
  NAUTILUS run (none has a newline left after it); stock Qwen has 67.
* **The VFE is switchable** through ``thuenen_pipeline/vfe_ablation.ConditionControl``,
  the same hooks as the Thuenen ablation, so the two ablations are one mechanism.
* Sharding over GPUs, append-only resume, and a paired image-cluster bootstrap
  of every ablation against ``full``.
* **Stock Qwen2.5-VL-7B-Instruct** (``--model qwen``, run dir ``stock_qwen<tag>``)
  goes down the same loop, so "does the fine-tune beat its own base model" is one
  more column. It is loaded by ``qwen_baseline.load_model`` (stock class, flash-attn
  2, bf16, NAUTILUS' ``generation_config`` copied on so the sampler path is the same)
  and differs in exactly two places: no ``double_image_tokens`` (one feature vector
  per patch, not two -- ``--self-test`` proves the count) and no ``ConditionControl``
  (no VFE; ``--condition full`` only). Prompt, ``process_vision_info`` path, tag-2/7
  rescale, ``max_new_tokens`` and record are untouched. Stock Qwen answers in its own
  format (fenced JSON, full sentences), which upstream's parsers mostly reject, so
  ``--report`` adds a per-task format-adherence block next to the scores.

Paper <-> upstream task names: coarse classification = ``Region_Classification``,
fine = ``Fishnet_Classification``; grounding PR@0.5 = ``Precision@0.5``; detection
mAP / mAP@0.5 = ``AP_50_95`` / ``AP_50``; counting acc = the choice questions'
``Accuracy``, MAE/RMSE = the numeric ones. ``logs/qwen_eval.json`` (the authors'
own metric dump) reproduces the paper's NAUTILUS(Qwen2.5-VL) row exactly.

Modes
-----
default        Inference for one ``--condition`` into
               ``<runs>/<condition><tag>/predictions[.shard{i}].jsonl``. Resumable:
               ids already in the file are skipped. ``--limit N`` takes N questions
               *per task*, chosen by sha1(id) order, so smaller limits nest in larger.
``--merge``    Fold shard files into ``predictions.json`` (upstream format and order)
               and shard metadata into ``metadata.json``.
``--score``    Upstream metrics per task -> ``metrics.json`` (``qwen_eval.json``'s shape).
``--degraded`` Grounding metrics per degraded subset -> ``metrics_degraded.json``.
               Both refuse an incomplete merge unless ``--allow-incomplete``.
``--report``   Every scored run dir under ``--runs`` with the base's question
               selection: paper value, ``qwen_eval.json``, each condition, deltas vs
               ``full`` with 95% paired bootstrap CIs (seeded per cell), plus format
               adherence per task (box answers upstream parses, bare option / int /
               letter, words per caption/VQA answer vs GT).
               Writes ``<runs>/report.{json,md}``.
``--self-test`` CPU, no weights: one tag-7 question through both checkpoints'
               processors -- identical text/ids/pixels, undoubled placeholders =
               ViT tokens, doubled = 2x -- and NAUTILUS_GENERATION == the shipped
               ``generation_config.json``.
``--diff-subset SMALL BIG``  Every id of SMALL byte-identical in BIG: the regression
               gate for a ``--limit`` run against a full one (``--diff`` wants
               equal id sets and stays strict).

Usage:
    # 0. smoke test: 5 questions per task, then score it
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --condition full --limit 5 --tag _smoke --device 0 && \\
       python3 nautdata_benchmark.py --condition full --tag _smoke --merge --score"

    # 1. hook gate: pass-through hooks must be byte-identical to no hooks
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --condition full --limit 5 --tag _smoke_noop \\
         --hook-noop --device 0 && \\
       python3 nautdata_benchmark.py --diff /workspace/runs/nautdata_benchmark/full_smoke \\
                                            /workspace/runs/nautdata_benchmark/full_smoke_noop"

    # 2. one condition over the full split, one shard per GPU (resumable). The log
    #    redirect needs the runs dir to exist before python3 starts; -d hides the error.
    docker exec nautilus-qwen mkdir -p /workspace/runs/nautdata_benchmark
    for i in 0 1 2 3; do docker exec -d nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --condition full --shard $i/4 --device $i \\
         > /workspace/runs/nautdata_benchmark/full.shard$i.log 2>&1"; done

    # 3. merge, score, Table 7, report (CPU)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --condition full --merge --score --degraded && \\
       python3 nautdata_benchmark.py --report"

    # 4. stock Qwen2.5-VL: gates, kept out of the real runs dir (smoke dir, beta 4090)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       R=/workspace/runs/nautdata_benchmark_smoke && \\
       python3 nautdata_benchmark.py --self-test && \\
       python3 nautdata_benchmark.py --condition full --limit 5 --tag _regress \\
         --device 0 --runs \\$R && \\
       python3 nautdata_benchmark.py --condition full --tag _regress --runs \\$R \\
         --merge --score --degraded && \\
       python3 nautdata_benchmark.py --diff-subset \\$R/full_regress \\
         /workspace/runs/nautdata_benchmark/full && \\
       python3 nautdata_benchmark.py --model qwen --limit 5 --device 0 --runs \\$R && \\
       python3 nautdata_benchmark.py --model qwen --runs \\$R --merge --score --degraded && \\
       python3 nautdata_benchmark.py --report --tag _regress --runs \\$R"

    # 5. stock Qwen2.5-VL over the full split (beta 4090s, same GPUs as full), then
    #    merge/score/Table 7 and the report, which picks stock_qwen up next to full
    for i in 0 1 2 3; do docker exec -d nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --model qwen --shard $i/4 --device $i \\
         > /workspace/runs/nautdata_benchmark/stock_qwen.shard$i.log 2>&1"; done
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/nautdata_pipeline && \\
       python3 nautdata_benchmark.py --model qwen --merge --score --degraded && \\
       python3 nautdata_benchmark.py --report"
"""

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
QWEN_ROOT = os.path.dirname(SCRIPTS)
NAUTILUS_ROOT = os.path.dirname(QWEN_ROOT)
EVAL_DIR = os.path.join(NAUTILUS_ROOT, "eval")
sys.path.insert(0, os.path.join(SCRIPTS, "thuenen_pipeline"))
sys.path.insert(0, EVAL_DIR)

import numpy as np  # noqa: E402

DEFAULT_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"
STOCK_CHECKPOINT = "/workspace/weights/qwen2.5-vl-7b-instruct"
# --model -> (default checkpoint, run dir stem). The stock model has no VFE, so it has
# exactly one condition and its own dir name instead of a condition name.
MODELS = {"nautilus": (DEFAULT_CHECKPOINT, None),
          "qwen": (STOCK_CHECKPOINT, "stock_qwen")}
DEFAULT_TEST_JSON = "/workspace/datasets/nautdata/nautilus-qwen-instruct-test.json"
DEFAULT_DEGRADED_DIR = "/workspace/datasets/nautdata"
DEFAULT_IMAGES = "/workspace/nautdata"
DEFAULT_RUNS = "/workspace/runs/nautdata_benchmark"
AUTHORS_EVAL = os.path.join(NAUTILUS_ROOT, "logs", "qwen_eval.json")
MAX_PIXEL = 1338          # eval4nautilus_qwen.py's --max-pixel default
MAX_NEW_TOKENS = 2048     # eval4nautilus_qwen.py's generate()
CONDITIONS = ["full", "no_gain", "no_backscatter", "vfe_identity",
              "dino_unscrambled", "dino_unscrambled_imagenet"]  # == vfe_ablation.CONDITIONS
DEGRADED_FILES = ["light_split.json", "color_split.json", "sharpness_split.json"]
DEGRADED_ORDER = ["low-light", "normal-light", "green-color", "blue-color",
                  "turbid", "clear"]
DEGRADED_ALIASES = {"USIS/": "USIS10K/", "UOT-400/": "UVOT-400/"}
BOOTSTRAP_REPS = 2000

# (label, task, path into metrics.json, paper value, higher is better)
# Paper values: Table 2 / Table 3 row "NAUTILUS (Qwen2.5-VL)", appendix Tables III-V.
TABLE2 = [
    ("coarse acc", "Region_Classification", ["Accuracy"], 0.903, True),
    ("fine acc", "Fishnet_Classification", ["Accuracy"], 0.938, True),
    ("image cap. METEOR", "Image_caption", ["meteor_score"], 0.223, True),
    ("image cap. CIDEr", "Image_caption", ["cider_score"], 1.023, True),
    ("region cap. METEOR", "Region_caption", ["meteor_score"], 0.199, True),
    ("region cap. CIDEr", "Region_caption", ["cider_score"], 1.258, True),
    ("grounding mIoU", "Grounding", ["mIoU"], 0.538, True),
    ("grounding PR@0.5", "Grounding", ["Precision@0.5"], 0.588, True),
    ("detection mAP", "Detection", ["AP_50_95"], 0.258, True),
    ("detection mAP@0.5", "Detection", ["AP_50"], 0.453, True),
    ("VQA METEOR", "VQA", ["meteor_score"], 0.381, True),
    ("counting MAE", "Counting", ["Counting Evaluation Metrics",
                                  "Mean Absolute Error (MAE)"], 30.9, False),
    ("counting RMSE", "Counting", ["Counting Evaluation Metrics",
                                   "Root Mean Squared Error (RMSE)"], 93.7, False),
    ("counting acc", "Counting", ["Choice Evaluation Metrics", "Accuracy"], 0.707, True),
]
# The TABLE2 labels ``per_question`` decomposes (ratio of sums). Only these get a
# bootstrap CI, and ``report`` requires every one of them to reproduce metrics.json.
# RMSE, METEOR, CIDEr and COCO mAP are not ratios of per-question sums.
DECOMPOSED = ("coarse acc", "fine acc", "grounding mIoU", "grounding PR@0.5",
              "counting MAE", "counting acc")
# Table 7, PR@0.5 per subset.
TABLE7 = {"low-light": 0.585, "normal-light": 0.587, "green-color": 0.577,
          "blue-color": 0.592, "turbid": 0.554, "clear": 0.608}


# ── upstream eval code ───────────────────────────────────────────────────────
def upstream():
    """``NAUTILUS/eval/utils.py`` under an unambiguous name (it is called ``utils``)."""
    return importlib.import_module("utils")


def score_kwargs(args):
    return {"model_type": "Qwen", "image_folder": args.images, "vqa_acc": False,
            "java": True}


# ── selection / plumbing ─────────────────────────────────────────────────────
def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path):
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shard(text):
    """``"i/N"`` -> ``(i, N)``; ``None`` -> ``(0, 1)``. Mirrors qwen_baseline.py."""
    if not text:
        return 0, 1
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise SystemExit("--shard must be i/N, got {!r}".format(text))
    if count < 1 or not 0 <= index < count:
        raise SystemExit("--shard must be i/N with 0 <= i < N, got {!r}".format(text))
    return index, count


def load_questions(args):
    with open(args.test_json, encoding="utf-8") as handle:
        return json.load(handle)


def select(args, questions):
    """Questions for this run: task filter, then ``--limit`` per task by sha1(id)."""
    tags = upstream().task_tags
    wanted = None
    if args.tasks:
        wanted = set()
        for name in args.tasks.split(","):
            name = name.strip()
            wanted.add(tags.get(name, name))
        unknown = wanted - set(tags.values())
        if unknown:
            raise SystemExit("unknown task(s) {}; use names or tags {}".format(
                sorted(unknown), tags))
    by_tag = defaultdict(list)
    for question in questions:
        if wanted is None or question["id"][1] in wanted:
            by_tag[question["id"][1]].append(question)
    chosen = []
    for tag in sorted(by_tag):
        bucket = by_tag[tag]
        if args.limit:
            bucket = sorted(bucket, key=lambda q: sha1(q["id"]))[:args.limit]
        chosen.extend(bucket)
    order = {q["id"]: i for i, q in enumerate(questions)}
    chosen.sort(key=lambda q: order[q["id"]])
    return chosen


def run_dir(args):
    stem = MODELS[args.model][1] or args.condition
    return os.path.join(args.runs, stem + args.tag)


def shard_suffix(index, count):
    """Always suffixed, even unsharded: ``metadata.json`` is ``--merge``'s output."""
    return ".shard{}".format(index)


def git_sha():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- dubious-ownership etc. inside the container
        return None


def code_sha1():
    """HEAD alone does not pin the code: the full runs were made at a HEAD where this
    script was still untracked. Hash the two files that define the run instead."""
    paths = {"nautdata_benchmark.py": os.path.abspath(__file__),
             "vfe_ablation.py": os.path.join(SCRIPTS, "thuenen_pipeline", "vfe_ablation.py"),
             "qwen_baseline.py": os.path.join(SCRIPTS, "thuenen_pipeline", "qwen_baseline.py")}
    return {name: file_sha1(path) for name, path in paths.items()}


# Metadata keys a resumed run, and every shard of a merge, must agree on. Runs made
# before ``model``/``double_tokens``/``generation_applied`` existed lack them, and a
# key only one side defines is not compared -- so those dirs stay resumable.
IDENTITY_KEYS = ("checkpoint", "model", "double_tokens", "generation_applied",
                 "condition", "hook_noop", "test_json_sha1",
                 "selection_sha1", "max_pixel", "max_new_tokens", "gpu_name")


def identity_mismatch(previous, current):
    """``[(key, previous, current)]`` for every identity key both define differently."""
    return [(k, previous[k], current[k]) for k in IDENTITY_KEYS
            if k in previous and k in current and previous[k] != current[k]]


def read_metadata(out_dir):
    path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(path):
        raise SystemExit("{}: no metadata.json, run --merge first".format(out_dir))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def require_complete(out_dir, args):
    """Refuse to score a merge that is missing predictions, unless asked not to.

    ``merge`` only flags an incomplete run; without this, ``--merge --score
    --degraded`` would carry on and publish numbers on a partial split.
    """
    meta = read_metadata(out_dir)
    n, expected = meta.get("n_predictions"), meta.get("selection_n")
    if n != expected and not args.allow_incomplete:
        raise SystemExit("{}: {} of {} predictions; finish inference or pass "
                         "--allow-incomplete".format(out_dir, n, expected))
    return meta


def is_partial_by_design(meta):
    """A ``--limit``/``--tasks`` run covers only part of each subset on purpose."""
    return bool(meta.get("limit_per_task")) or bool(meta.get("tasks"))


def read_jsonl(path):
    records = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ── inference ────────────────────────────────────────────────────────────────
def resolve_model(args):
    """Fill the per-model checkpoint default and refuse combinations that mean nothing.

    Stock Qwen has no VFE: no condition but ``full``, no hooks, and its own run dir.
    A NAUTILUS checkpoint under ``--model qwen`` (or the reverse) would load the wrong
    class on the right weights, so the two default paths are not interchangeable.
    """
    if args.checkpoint is None:
        args.checkpoint = MODELS[args.model][0]
    if args.model == "qwen":
        if args.condition != "full" or args.hook_noop:
            raise SystemExit("--model qwen has no VFE: use --condition full, no --hook-noop")
        if os.path.realpath(args.checkpoint) == os.path.realpath(DEFAULT_CHECKPOINT):
            raise SystemExit("--model qwen with the NAUTILUS checkpoint")
    elif os.path.realpath(args.checkpoint) == os.path.realpath(STOCK_CHECKPOINT):
        raise SystemExit("--model nautilus with the stock checkpoint; use --model qwen")


def build_inputs(data, args, processor, utils):
    """``eval4nautilus_qwen.eval_with_infer`` up to the token doubling, verbatim.

    Shared by ``infer`` and ``--self-test``, so the gate checks the path that runs.
    """
    from qwen_vl_utils import process_vision_info

    image_processor = processor.image_processor
    image_path = os.path.join(args.images, data["image"])
    prompt = data["conversations"][0]["value"].replace("<image>", "").strip()
    if data["id"][1] in ["2", "7"]:
        grid_thw, ori_w, ori_h = utils.get_grid_thw(image_processor, image_path)
        i_height = grid_thw[1].item() * 14
        i_width = grid_thw[2].item() * 14
        scale_h, scale_w = i_height / ori_h, i_width / ori_w
        prompt = utils.scale_bboxes_in_text(prompt, scale_w, scale_h)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14
    return prompt, text, inputs, input_height, input_width


def infer(args):
    import torch
    from tqdm import tqdm

    import vfe_ablation
    from vfe_ablation import ConditionControl, load_model

    if CONDITIONS != vfe_ablation.CONDITIONS:
        raise SystemExit("CONDITIONS here and in vfe_ablation.py have drifted apart")
    stock = args.model == "qwen"
    if stock:
        from qwen_baseline import NAUTILUS_GENERATION
        from qwen_baseline import load_model as load_stock_model
    utils = upstream()
    shard_index, shard_count = parse_shard(args.shard)
    questions = select(args, load_questions(args))
    mine = questions[shard_index::shard_count]
    out_dir = run_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    suffix = shard_suffix(shard_index, shard_count)
    pred_path = os.path.join(out_dir, "predictions{}.jsonl".format(suffix))
    meta_path = os.path.join(out_dir, "metadata{}.json".format(suffix))

    # Every shard file in the dir, not only this shard's: a resume with a different
    # shard count must not redo questions another shard file already holds (merge
    # would then abort on the duplicate after the GPU time is spent).
    done = set()
    for name in os.listdir(out_dir):
        if name.startswith("predictions") and name.endswith(".jsonl"):
            done.update(r["id"] for r in read_jsonl(os.path.join(out_dir, name)))
    stray = done - {q["id"] for q in questions}
    if stray:
        raise SystemExit("{}: {} predictions on disk lie outside this selection (a smaller "
                         "--limit or other --tasks?); use a new --tag".format(
                             out_dir, len(stray)))
    todo = [q for q in mine if q["id"] not in done]
    print("[{}] {} questions in shard, {} done, {} to go".format(
        os.path.basename(out_dir), len(mine), len(mine) - len(todo), len(todo)))
    if not todo:
        return

    previous = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            previous = json.load(handle)

    gpu_name = torch.cuda.get_device_name(args.device)
    metadata = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        # The VFE emits two feature vectors per patch; stock Qwen one (see --self-test).
        "double_tokens": not stock,
        # Stock Qwen gets NAUTILUS' generation_config so the sampler path is identical;
        # NAUTILUS decodes with its checkpoint's own (null = nothing overridden).
        "generation_applied": dict(NAUTILUS_GENERATION) if stock else None,
        "condition": args.condition,
        "hook_noop": args.hook_noop,
        "test_json": args.test_json,
        "test_json_sha1": file_sha1(args.test_json),
        "selection_n": len(questions),
        "selection_sha1": sha1("\n".join(sorted(q["id"] for q in questions))),
        "limit_per_task": args.limit,
        "tasks": args.tasks,
        "shard": "{}/{}".format(shard_index, shard_count),
        "shard_n": len(mine),
        "max_pixel": MAX_PIXEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gpu_name": gpu_name,
        "device": "cuda:{}".format(args.device),
        "git_sha": git_sha(),
        "code_sha1": code_sha1(),
        "torch": torch.__version__,
        "transformers": importlib.import_module("transformers").__version__,
        "known_defects": [
            "upstream get_grid_thw preprocesses the original image, not the "
            "28-rounded one the model is fed; tag 2/7 prompt boxes are rescaled "
            "with that grid. Kept on purpose: it is the path the paper's numbers "
            "were produced with."],
        "seconds": previous.get("seconds", 0.0),
    }
    # Every metadata file already in the dir (other shards, an earlier merge) must
    # describe the same run. The dir name carries condition + tag but not the
    # checkpoint, so without this a new --checkpoint would append to the old file.
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("metadata") and name.endswith(".json"):
            with open(os.path.join(out_dir, name), encoding="utf-8") as handle:
                clash = identity_mismatch(json.load(handle), metadata)
            if clash:
                raise SystemExit("{}/{} describes a different run: {}. Greedy outputs "
                                 "would be mixed; use a new --tag.".format(
                                     out_dir, name, clash))
    # Written before the first question too, so a killed run still leaves the guard.
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(dict(metadata, n_written_total=len(mine) - len(todo)), handle, indent=2)

    control = None
    if stock:
        model, processor, applied = load_stock_model(argparse.Namespace(
            checkpoint=args.checkpoint, device=args.device, query_max_pixels=MAX_PIXEL))
        if applied != metadata["generation_applied"]:
            raise SystemExit("generation overrides {} != metadata {}".format(
                applied, metadata["generation_applied"]))
    else:
        model, processor = load_model(args.checkpoint, args.device, MAX_PIXEL)
        control = ConditionControl(model.visual)
        control.apply(args.condition, hook_noop=args.hook_noop)
    started = time.time()
    stats = Counter()
    try:
        with open(pred_path, "a", encoding="utf-8") as sink:
            for data in tqdm(todo, desc=os.path.basename(out_dir)):
                # ── eval4nautilus_qwen.eval_with_infer, verbatim ──
                prompt, _text, inputs, input_height, input_width = build_inputs(
                    data, args, processor, utils)
                if metadata["double_tokens"]:
                    inputs["input_ids"], inputs["attention_mask"] = \
                        utils.double_image_tokens(inputs, utils.image_token_id)
                inputs = inputs.to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                trimmed = [out[len(inp):] for inp, out in
                           zip(inputs.input_ids, generated_ids)]
                res_text = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record = {"id": data["id"], "gt": data["conversations"][1]["value"],
                          "text": res_text, "prompt": prompt,
                          "input_height": input_height.item(),
                          "input_width": input_width.item()}
                # ── end verbatim ──
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                stats["written"] += 1
    finally:
        if control is not None:
            control.clear()
        metadata["seconds"] += time.time() - started
        metadata["n_written_total"] = len(mine) - len(todo) + stats["written"]
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    print("[done] {} written, {:.2f} s/question".format(
        stats["written"], (time.time() - started) / max(stats["written"], 1)))


# ── --self-test ──────────────────────────────────────────────────────────────
def self_test(args):
    """CPU, no weights: the stock and NAUTILUS inputs differ in the doubling only.

    Builds one tag-7 question (it goes through the prompt-box rescale too) through
    ``build_inputs`` with each checkpoint's own processor. Both must produce the same
    chat text, ids, grid and pixels; undoubled placeholders must equal the ViT tokens
    the LLM is handed (``t*h*w / merge**2``), doubled ones twice that. Also checks that
    the overrides copied onto stock Qwen are NAUTILUS' ``generation_config.json``.
    """
    import torch
    from transformers import AutoProcessor

    from qwen_baseline import NAUTILUS_GENERATION

    utils = upstream()
    pool = [q for q in load_questions(args)
            if q["id"][1] == utils.task_tags["Region_Classification"]]
    data = min(pool, key=lambda q: sha1(q["id"]))
    built = {}
    for model, (checkpoint, _) in MODELS.items():
        processor = AutoProcessor.from_pretrained(
            checkpoint, min_pixels=1 * 28 * 28, max_pixels=MAX_PIXEL * 28 * 28)
        prompt, text, inputs, _, _ = build_inputs(data, args, processor, utils)
        built[model] = (prompt, text, inputs, processor.image_processor.merge_size)
    failures = []
    for part, index in (("prompt", 0), ("chat text", 1)):
        if built["qwen"][index] != built["nautilus"][index]:
            failures.append("{} differs between the two processors".format(part))
    for key in ("input_ids", "image_grid_thw", "pixel_values"):
        if not torch.equal(built["qwen"][2][key], built["nautilus"][2][key]):
            failures.append("{} differs between the two processors".format(key))

    inputs, merge_size = built["qwen"][2], built["qwen"][3]
    grid = inputs["image_grid_thw"][0]
    expected = int(grid[0] * grid[1] * grid[2]) // (merge_size * merge_size)
    plain = int((inputs["input_ids"][0] == utils.image_token_id).sum())
    doubled_ids, _ = utils.double_image_tokens(built["nautilus"][2], utils.image_token_id)
    doubled = int((doubled_ids[0] == utils.image_token_id).sum())
    print("{}  {}  grid {}x{}x{}  merge {}".format(
        data["id"], data["image"], int(grid[0]), int(grid[1]), int(grid[2]), merge_size))
    print("  ViT tokens the LLM is handed          {}".format(expected))
    print("  placeholders, stock (no doubling)     {}".format(plain))
    print("  placeholders, NAUTILUS (doubled)      {}".format(doubled))
    if plain != expected:
        failures.append("stock placeholders {} != ViT tokens {}".format(plain, expected))
    if doubled != 2 * expected:
        failures.append("NAUTILUS placeholders {} != 2x {}".format(doubled, expected))

    with open(os.path.join(DEFAULT_CHECKPOINT, "generation_config.json"),
              encoding="utf-8") as handle:
        shipped = json.load(handle)
    wrong = {k: (v, shipped.get(k)) for k, v in NAUTILUS_GENERATION.items()
             if shipped.get(k) != v}
    if wrong:
        failures.append("NAUTILUS_GENERATION vs generation_config.json: {}".format(wrong))
    for failure in failures:
        print("  [FAIL] {}".format(failure))
    if failures:
        raise SystemExit(1)
    print("  [ok] identical inputs up to the doubling; NAUTILUS_GENERATION matches the "
          "checkpoint")


# ── merge / score ────────────────────────────────────────────────────────────
def merge(args):
    out_dir = run_dir(args)
    files = sorted(f for f in os.listdir(out_dir)
                   if f.startswith("predictions") and f.endswith(".jsonl"))
    records = {}
    for name in files:
        for record in read_jsonl(os.path.join(out_dir, name)):
            if record["id"] in records:
                raise SystemExit("{}: id {} appears twice".format(out_dir, record["id"]))
            records[record["id"]] = record
    metas = sorted(f for f in os.listdir(out_dir)
                   if f.startswith("metadata") and f.endswith(".json")
                   and f != "metadata.json")
    merged = None
    for name in metas:
        if "predictions{}l".format(name[len("metadata"):]) not in files:
            print("[merge] {}: {} has no predictions file, ignored".format(out_dir, name))
            continue
        with open(os.path.join(out_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        if merged is None:
            merged = dict(payload, seconds=0.0, shards=[])
        clash = identity_mismatch(merged, payload)
        if clash:
            raise SystemExit("{}: shards describe different runs: {}".format(out_dir, clash))
        merged["seconds"] += payload["seconds"]
        merged["shards"].append({k: payload[k] for k in ("shard", "shard_n", "device")})
    if merged is None and os.path.isfile(os.path.join(out_dir, "metadata.json")):
        with open(os.path.join(out_dir, "metadata.json"), encoding="utf-8") as handle:
            merged = json.load(handle)
    if merged is None:
        raise SystemExit("{}: no metadata".format(out_dir))
    for key in ("shard", "shard_n", "device", "n_written_total"):
        merged.pop(key, None)

    order = {q["id"]: i for i, q in enumerate(load_questions(args))}
    predictions = sorted(records.values(), key=lambda r: order[r["id"]])
    merged["n_predictions"] = len(predictions)
    flag = "" if len(predictions) == merged["selection_n"] else \
        "   <-- INCOMPLETE, expected {}".format(merged["selection_n"])
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    print("[merge] {}: {} predictions{}".format(out_dir, len(predictions), flag))


def load_predictions(out_dir):
    path = os.path.join(out_dir, "predictions.json")
    if not os.path.isfile(path):
        raise SystemExit("{}: run --merge first".format(out_dir))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def questions_for(predictions, questions):
    ids = {p["id"] for p in predictions}
    return [q for q in questions if q["id"] in ids]


TEXT_TASKS = ("Image_caption", "Region_caption", "VQA")


def flatten_newlines(task_answers):
    """Newline runs -> one space, so METEOR's one-request-per-line protocol holds.

    Returns new records (the caller's predictions are not mutated) and how many
    changed *after* upstream's ``process_text``, i.e. how many would have reached
    METEOR with a newline in them.
    """
    out, changed = [], 0
    for answer in task_answers:
        text = re.sub(r"[\r\n]+", " ", answer["text"])
        if text != answer["text"]:
            changed += re.sub(r"[\W_]+$", "", answer["text"]).strip() != \
                re.sub(r"[\W_]+$", "", text).strip()
            answer = dict(answer, text=text)
        out.append(answer)
    return out, changed


def score(args):
    utils = upstream()
    out_dir = run_dir(args)
    require_complete(out_dir, args)
    predictions = load_predictions(out_dir)
    questions = questions_for(predictions, load_questions(args))
    answers, truths = utils.sortbyid(predictions), utils.sortbyid(questions)
    metrics = {}
    for task, task_answers in answers.items():
        print("[score] {} ({} questions)".format(task, len(task_answers)))
        if task in TEXT_TASKS:
            task_answers, flattened = flatten_newlines(task_answers)
            if flattened:
                print("[score] {}: {} answers had newlines, joined with spaces "
                      "for METEOR".format(task, flattened))
        try:
            metrics[task] = utils.Metric_method_dict[task](task_answers, truths[task],
                                                            score_kwargs(args))
        except IndexError:
            # Upstream detection hands pycocotools an empty prediction list when no
            # answer parses (stock Qwen fences every list in ```json), and loadRes
            # indexes [0]. Upstream grounding catches the same case and returns {};
            # record it in upstream's counting "Error" convention instead of a number.
            if task != "Detection":
                raise
            from Evaluation_pack.detection import parse_bboxes_qwen
            with contextlib.redirect_stdout(io.StringIO()):
                parsed = sum(parse_bboxes_qwen(a["text"]) is not None for a in task_answers)
            metrics[task] = {"Error": "no prediction box reached COCO (pycocotools "
                                      "loadRes on an empty list)",
                             "n_parsed": parsed, "n_questions": len(task_answers)}
            print("[score] Detection: {} of {} answers parse, no box scored".format(
                parsed, len(task_answers)))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    for label, task, path, paper, _ in TABLE2:
        value = dig(metrics.get(task), path)
        if value is not None:
            print("  {:<20} {:>9.4f}   paper {}".format(label, value, paper))


def degraded_subsets(args):
    """Subset -> image paths, in the test JSON's directory names.

    The split files name two source datasets differently from the test JSON
    (``USIS/`` vs ``USIS10K/``, ``UOT-400/`` vs ``UVOT-400/``). Unaliased, those 186
    of the 750 images match no question and silently drop out of every subset.
    """
    subsets = {}
    for name in DEGRADED_FILES:
        with open(os.path.join(args.degraded_dir, name), encoding="utf-8") as handle:
            for subset, images in json.load(handle).items():
                subsets[subset] = {alias_image(image) for image in images}
    return subsets


def alias_image(path):
    for old, new in DEGRADED_ALIASES.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def degraded(args):
    utils = upstream()
    out_dir = run_dir(args)
    partial = is_partial_by_design(require_complete(out_dir, args))
    predictions = load_predictions(out_dir)
    grounding = [q for q in questions_for(predictions, load_questions(args))
                 if q["id"][1] == utils.task_tags["Grounding"]]
    pred_by_id = {p["id"]: p for p in predictions}
    result = {}
    for subset, images in degraded_subsets(args).items():
        truth = [q for q in grounding if q["image"] in images]
        answers = [pred_by_id[q["id"]] for q in truth]
        # Checked before the empty-subset skip: a full run in which a whole subset
        # matches nothing (a broken alias) must fail, not drop the row.
        n_images = len({q["image"] for q in truth})
        if n_images != len(images) and not partial:
            raise SystemExit("[degraded] {}: {} of {} split images matched a grounding "
                             "question".format(subset, n_images, len(images)))
        if not truth:
            print("[degraded] {}: no grounding questions in this run".format(subset))
            continue
        metrics = utils.evaluate_ground_metrics(answers, truth, score_kwargs(args))
        metrics["n_questions"] = len(truth)
        metrics["n_images"] = n_images
        metrics["n_images_in_split_file"] = len(images)
        result[subset] = metrics
        print("[degraded] {:<13} n={:<4} PR@0.5 {:.4f}   paper {}".format(
            subset, len(truth), metrics["Precision@0.5"], TABLE7[subset]))
    with open(os.path.join(out_dir, "metrics_degraded.json"), "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2)


def dig(obj, path):
    """``obj[path[0]][path[1]]...``, or ``None`` where upstream left the key out.

    Upstream's counting scorer writes ``{"Error": ...}`` in place of MAE/RMSE/Accuracy
    when a bucket is empty (a small ``--limit``, or a run that never answers with a
    bare int), so a missing key is a legitimate outcome, not a crash.
    """
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


# ── per-question scores (for the bootstrap) ──────────────────────────────────
def per_question(args, predictions, questions):
    """``{metric label: {id: (numerator, denominator)}}``, built from upstream's own
    parsers so that ratio-of-sums over all ids reproduces ``metrics.json`` exactly
    (asserted in ``report``).
    """
    import torch
    from PIL import Image
    from torchvision.ops import box_iou

    utils = upstream()
    from Evaluation_pack.count import build_ground_truth_map, build_prediction_map
    from Evaluation_pack.grounding import parse_bboxes_qwen

    tags = utils.task_tags
    q_by_id = {q["id"]: q for q in questions}
    out = defaultdict(dict)
    sizes = {}

    count_preds = [p for p in predictions if p["id"][1] == tags["Counting"]]
    gt_map = build_ground_truth_map([q_by_id[p["id"]] for p in count_preds])
    pred_map = build_prediction_map(count_preds)

    for pred in predictions:
        qid, tag = pred["id"], pred["id"][1]
        question = q_by_id[qid]
        if tag in (tags["Region_Classification"], tags["Fishnet_Classification"]):
            truth = question["conversations"][-1]["value"].strip().lower()
            guess = pred["text"].strip().lower()
            label = "coarse acc" if tag == tags["Region_Classification"] else "fine acc"
            out[label][qid] = (float(truth in guess), 1.0)
        elif tag == tags["Counting"]:
            g, p = gt_map[qid], pred_map[qid]
            if g["type"] != p["type"] or g["answer"] is None or p["answer"] is None:
                continue
            if g["type"] == "choice":
                out["counting acc"][qid] = (float(g["answer"] == p["answer"]), 1.0)
            else:
                out["counting MAE"][qid] = (float(abs(p["answer"] - g["answer"])), 1.0)
        elif tag == tags["Grounding"]:
            gt = parse_bboxes_qwen(question["conversations"][1].get("value", ""))
            if not gt:
                continue
            gt_boxes = torch.tensor(gt["default"], dtype=torch.float32)
            parsed = parse_bboxes_qwen(pred.get("text", ""))
            boxes = []
            if parsed:
                if question["image"] not in sizes:
                    with Image.open(os.path.join(args.images, question["image"])) as im:
                        sizes[question["image"]] = im.size
                w, h = sizes[question["image"]]
                for b in parsed["default"]:
                    if len(b) == 4:
                        boxes.append([b[0] * w / pred["input_width"],
                                      b[1] * h / pred["input_height"],
                                      b[2] * w / pred["input_width"],
                                      b[3] * h / pred["input_height"]])
            if not boxes:
                out["grounding PR@0.5"][qid] = (0.0, 0.0)
                out["grounding mIoU"][qid] = (0.0, 1.0)
                continue
            ious = box_iou(torch.tensor(boxes, dtype=torch.float32), gt_boxes)
            miou = ious.max(dim=0)[0].mean().item()
            if not np.isnan(miou):
                out["grounding mIoU"][qid] = (miou, 1.0)
            tp = float((ious.max(dim=1)[0] >= 0.5).sum().item())
            out["grounding PR@0.5"][qid] = (tp, float(len(boxes)))
    return out


def paired_bootstrap(base, other, cluster_of, seed_key):
    """Ratio-of-sums delta ``other - base`` over shared ids, resampling images.

    Seeded per cell from ``seed_key``, so a cell's CI does not move when another run
    dir is added or the rows are reordered (one shared generator made it do both).
    """
    rng = np.random.default_rng(int(sha1(seed_key)[:16], 16))
    ids = sorted(set(base) & set(other))
    clusters = defaultdict(list)
    for qid in ids:
        clusters[cluster_of[qid]].append(qid)
    keys = sorted(clusters)
    arr = lambda d, i: np.array([sum(d[q][i] for q in clusters[k]) for k in keys])  # noqa: E731
    bn, bd, on, od = arr(base, 0), arr(base, 1), arr(other, 0), arr(other, 1)
    ratio = lambda n, d: n.sum() / d.sum() if d.sum() else float("nan")  # noqa: E731
    point = ratio(on, od) - ratio(bn, bd)
    draws = rng.integers(0, len(keys), size=(BOOTSTRAP_REPS, len(keys)))
    deltas = np.array([ratio(on[d], od[d]) - ratio(bn[d], bd[d]) for d in draws])
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return point, lo, hi, len(ids)


# ── format adherence ─────────────────────────────────────────────────────────
def unfence(text, unwrap_list):
    """Strip a Markdown code fence, and with ``unwrap_list`` a JSON array's brackets.

    Stock Qwen answers box requests as a fenced JSON array; NautData's GT (and
    NAUTILUS) is bare objects, one per line, which upstream detection wraps in ``[]``
    itself. Only for the adherence diagnostic -- the scored text is never touched.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        body = body.strip()
    if unwrap_list and body.startswith("[") and body.endswith("]"):
        body = body[1:-1].strip()
    return body


def adherence(predictions, questions):
    """``[(task, measure, gt value or None, run value, n)]``: does the answer have the
    shape upstream's scorer can use?

    A model that answers right in the wrong shape scores as wrong, and stock Qwen
    was never trained on NautData's answer formats. "Parsed" uses upstream's own
    parsers (grounding's falls back to a ``[0, 0, 0, 0]`` box on any exception, which
    is what it scores, so that sentinel counts as rejected). Captions and VQA have no
    format to fail, but CIDEr punishes length, so their row is words per answer.
    """
    from Evaluation_pack import detection as up_detection
    from Evaluation_pack import grounding as up_grounding

    tags = upstream().task_tags
    q_by_id = {q["id"]: q for q in questions}
    by_tag = defaultdict(list)
    for pred in predictions:
        by_tag[pred["id"][1]].append(pred)
    share = lambda hits: (sum(hits) / len(hits), len(hits)) if hits else (None, 0)  # noqa: E731
    rows = []
    ground_ok = lambda g: g is not None and g["default"] != [[0.0, 0.0, 0.0, 0.0]]  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints every rejection
        for task, parse, accepted, unwrap in (
                ("Grounding", up_grounding.parse_bboxes_qwen, ground_ok, False),
                ("Detection", up_detection.parse_bboxes_qwen, lambda d: d is not None, True)):
            texts = [p["text"] for p in by_tag[tags[task]]]
            rows.append((task, "parsed by upstream", None) + share(
                [accepted(parse(t)) for t in texts]))
            # Diagnostic only, never scored: how much of the failure is the fence.
            rows.append((task, "parsed once unfenced", None) + share(
                [accepted(parse(unfence(t, unwrap))) for t in texts]))
    for task in ("Region_Classification", "Fishnet_Classification"):
        hits, several = [], []
        for pred in by_tag[tags[task]]:
            offered = q_by_id[pred["id"]]["conversations"][0]["value"].split(
                "as the answer:")[-1]
            options = {o.strip().lower().rstrip(".") for o in offered.split(",")}
            guess = pred["text"].strip().lower()
            hits.append(guess.rstrip(".") in options)
            # Upstream scores ``truth in guess``: an answer naming several options
            # is credited if any of them is right. Whole words, so "human divers"
            # is one option, not also "diver", and "starfish" is not "fish".
            named = [o for o in options if re.search(
                r"(?<![\w/]){}(?![\w/])".format(re.escape(o)), guess)]
            several.append(len(named) >= 2)
        rows.append((task, "exactly one offered option") + (None,) + share(hits))
        rows.append((task, "names >=2 offered options") + (None,) + share(several))
    numeric, choice = [], []
    for pred in by_tag[tags["Counting"]]:
        if pred["gt"].lower() in "abcd":  # upstream's own choice/numeric split
            choice.append(pred["text"].lower() in ("a", "b", "c", "d"))
        else:
            try:
                int(pred["text"].lower())
                numeric.append(True)
            except ValueError:
                numeric.append(False)
    rows.append(("Counting", "numeric: bare int", None) + share(numeric))
    rows.append(("Counting", "choice: bare letter", None) + share(choice))
    for task in ("Image_caption", "Region_caption", "VQA"):
        preds = by_tag[tags[task]]
        if not preds:
            rows.append((task, "words per answer", None, None, 0))
            continue
        words = lambda s: len(s.split())  # noqa: E731
        gt = np.mean([words(q_by_id[p["id"]]["conversations"][1]["value"]) for p in preds])
        rows.append((task, "words per answer", float(gt),
                     float(np.mean([words(p["text"]) for p in preds])), len(preds)))
    return rows


# ── report ───────────────────────────────────────────────────────────────────
def report(args):
    questions = load_questions(args)
    q_by_id = {q["id"]: q for q in questions}
    cluster_of = {qid: q["image"] for qid, q in q_by_id.items()}
    with open(AUTHORS_EVAL, encoding="utf-8") as handle:
        authors = json.load(handle)

    runs = {}
    for name in sorted(os.listdir(args.runs)):
        out_dir = os.path.join(args.runs, name)
        if os.path.isfile(os.path.join(out_dir, "metrics.json")):
            runs[name] = out_dir
    if not runs:
        raise SystemExit("no scored runs under {}".format(args.runs))
    base_name = "full" + args.tag
    ordered = ([base_name] if base_name in runs else []) + \
        [n for n in runs if n != base_name]
    metas = {name: read_metadata(runs[name]) for name in ordered}
    # One table, one question set: a smoke or --limit dir next to the full runs would
    # otherwise get a column, and a delta over a handful of shared ids beside it.
    if base_name in metas:
        selection = metas[base_name].get("selection_sha1")
        skipped = [n for n in ordered if metas[n].get("selection_sha1") != selection]
        for name in skipped:
            print("[report] {}: different question selection than {}, left out".format(
                name, base_name))
        ordered = [n for n in ordered if n not in skipped]

    loaded = {}
    gated = {}
    for name in ordered:
        with open(os.path.join(runs[name], "metrics.json"), encoding="utf-8") as handle:
            metrics = json.load(handle)
        degraded_path = os.path.join(runs[name], "metrics_degraded.json")
        deg = {}
        if os.path.isfile(degraded_path):
            with open(degraded_path, encoding="utf-8") as handle:
                deg = json.load(handle)
        predictions = load_predictions(runs[name])
        per = per_question(args, predictions, questions_for(predictions, questions))
        # Gate: the per-question decomposition must reproduce upstream's numbers --
        # for every decomposed metric upstream reports, so a label per_question
        # silently failed to produce cannot pass by being absent.
        gated[name] = []
        for label, task, path, _, _ in TABLE2:
            theirs = dig(metrics.get(task), path)
            if label not in DECOMPOSED or theirs is None:
                continue
            if label not in per:
                raise SystemExit("{}: metrics.json has {} but per_question produced no "
                                 "entries for it".format(name, label))
            n = sum(v[0] for v in per[label].values())
            d = sum(v[1] for v in per[label].values())
            mine = n / d if d else 0.0
            if abs(mine - theirs) > 1e-6:
                raise SystemExit("{}: per-question {} = {} but metrics.json says {}"
                                 .format(name, label, mine, theirs))
            gated[name].append(label)
        loaded[name] = {"metrics": metrics, "degraded": deg, "meta": metas[name], "per": per,
                        "adherence": adherence(predictions, questions_for(predictions,
                                                                          questions))}

    rows = []
    for label, task, path, paper, higher in TABLE2:
        row = {"metric": label, "paper": paper, "higher_is_better": higher,
               "authors_qwen_eval": dig(authors.get(task), path)}
        for name in ordered:
            row[name] = dig(loaded[name]["metrics"].get(task), path)
            if name != base_name and base_name in loaded and label in loaded[name]["per"] \
                    and label in loaded[base_name]["per"]:
                point, lo, hi, n = paired_bootstrap(loaded[base_name]["per"][label],
                                                    loaded[name]["per"][label],
                                                    cluster_of, label + "|" + name)
                row[name + " delta"] = [point, lo, hi, n]
        rows.append(row)
    subsets = degraded_subsets(args)
    for subset in DEGRADED_ORDER:
        label = "T7 PR@0.5 " + subset
        row = {"metric": label, "paper": TABLE7[subset],
               "higher_is_better": True, "authors_qwen_eval": None}
        for name in ordered:
            deg = loaded[name]["degraded"]
            row[name] = deg[subset]["Precision@0.5"] if subset in deg else None
        if base_name in loaded:
            images = subsets[subset]
            base = {q: v for q, v in loaded[base_name]["per"].get(
                "grounding PR@0.5", {}).items() if cluster_of[q] in images}
            for name in ordered:
                if name == base_name:
                    continue
                other = {q: v for q, v in loaded[name]["per"].get(
                    "grounding PR@0.5", {}).items() if cluster_of[q] in images}
                if base and other:
                    row[name + " delta"] = list(paired_bootstrap(
                        base, other, cluster_of, label + "|" + name))
        rows.append(row)

    adherence_rows = []
    for i, (task, measure, gt, _, _) in enumerate(loaded[ordered[0]]["adherence"]):
        row = {"task": task, "measure": measure, "gt": gt}
        for name in ordered:
            _, _, _, value, n = loaded[name]["adherence"][i]
            row[name] = value
            row[name + " n"] = n
        adherence_rows.append(row)

    payload = {"base": base_name, "runs": ordered, "rows": rows,
               "adherence": adherence_rows, "gated": gated,
               "gpu": {n: loaded[n]["meta"].get("gpu_name") for n in ordered},
               "n_predictions": {n: loaded[n]["meta"].get("n_predictions") for n in ordered},
               "bootstrap": "paired, image-cluster, {} reps, 95% percentile CI, "
                            "seeded per cell; delta = run - base in the metric's own "
                            "units".format(BOOTSTRAP_REPS)}
    with open(os.path.join(args.runs, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fmt = lambda v: "--" if v is None else ("{:.1f}".format(v) if abs(v) >= 10  # noqa: E731
                                            else "{:.3f}".format(v))
    header = ["metric", "paper", "qwen_eval.json"] + ordered
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        metric = row["metric"] + ("" if row["higher_is_better"] else " (lower is better)")
        cells = [metric, fmt(row["paper"]), fmt(row["authors_qwen_eval"])]
        for name in ordered:
            cell = fmt(row.get(name))
            delta = row.get(name + " delta")
            if delta:
                cell += " ({:+.3f} [{:+.3f}, {:+.3f}])".format(*delta[:3])
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Deltas vs `{}`: {}. n predictions: {}. GPUs: {}.".format(
        base_name, payload["bootstrap"], payload["n_predictions"], payload["gpu"]))
    lines.append("")
    lines.append("Format adherence (share of answers in the shape upstream's scorer "
                 "uses; captions/VQA: words per answer vs GT):")
    lines.append("")
    header = ["task", "measure", "GT"] + ordered
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in adherence_rows:
        cells = [row["task"], row["measure"], fmt(row["gt"])]
        cells += ["{} (n={})".format(fmt(row[name]), row[name + " n"]) for name in ordered]
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(args.runs, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def diff(args):
    a, b = (load_predictions(d) for d in args.diff)
    a, b = {p["id"]: p["text"] for p in a}, {p["id"]: p["text"] for p in b}
    shared = sorted(set(a) & set(b))
    same = sum(a[i] == b[i] for i in shared)
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    print("{} shared ids, {} byte-identical, {} differ; {} only in A, {} only in B".format(
        len(shared), same, len(shared) - same, only_a, only_b))
    for qid in [i for i in shared if a[i] != b[i]][:5]:
        print("  {}:\n    A {!r}\n    B {!r}".format(qid, a[qid][:200], b[qid][:200]))
    # The hook gate: an incomplete run must not pass on the few ids it does share.
    if same != len(shared) or only_a or only_b or not shared:
        raise SystemExit(1)


def diff_subset(args):
    """Regression gate: every id of the small run A, byte-identical in the big run B.

    ``--diff`` demands equal id sets, which a ``--limit`` run against a full split
    cannot meet. Here A must be non-empty and wholly contained in B; B's extra ids are
    the point and do not count.
    """
    small, big = (load_predictions(d) for d in args.diff_subset)
    small = {p["id"]: p["text"] for p in small}
    big = {p["id"]: p["text"] for p in big}
    missing = sorted(set(small) - set(big))
    shared = sorted(set(small) & set(big))
    differ = [i for i in shared if small[i] != big[i]]
    print("{} ids in A: {} byte-identical in B, {} differ, {} missing from B".format(
        len(small), len(shared) - len(differ), len(differ), len(missing)))
    by_task = Counter(i[1] for i in differ)
    if by_task:
        print("  differing ids per task tag: {}".format(dict(sorted(by_task.items()))))
    for qid in differ[:5]:
        print("  {}:\n    A {!r}\n    B {!r}".format(qid, small[qid][:200], big[qid][:200]))
    if differ or missing or not small:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="nautilus", choices=sorted(MODELS),
                        help="qwen = stock Qwen2.5-VL-7B-Instruct, run dir stock_qwen<tag>")
    parser.add_argument("--condition", default="full", choices=CONDITIONS)
    parser.add_argument("--tag", default="", help="suffix of the run dir name")
    parser.add_argument("--checkpoint", default=None,
                        help="default: the --model's own ({})".format(
                            ", ".join("{}={}".format(m, c) for m, (c, _) in
                                      sorted(MODELS.items()))))
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--degraded-dir", default=DEFAULT_DEGRADED_DIR)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard", default=None, help="i/N")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task names or tags (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="questions per task, by sha1(id) order (nested)")
    parser.add_argument("--hook-noop", action="store_true",
                        help="pass-through hooks; must equal an unhooked full run")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="--score/--degraded a merge with missing predictions")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--diff", nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--diff-subset", nargs=2, metavar=("SMALL", "BIG"),
                        help="every id of SMALL must be byte-identical in BIG")
    parser.add_argument("--self-test", action="store_true",
                        help="CPU: stock vs NAUTILUS inputs differ only in the doubling")
    args = parser.parse_args()
    resolve_model(args)

    if args.self_test:
        return self_test(args)
    if args.diff:
        return diff(args)
    if args.diff_subset:
        return diff_subset(args)
    if args.report:
        return report(args)
    if args.merge or args.score or args.degraded:
        if args.merge:
            merge(args)
        if args.score:
            score(args)
        if args.degraded:
            degraded(args)
        return None
    return infer(args)


if __name__ == "__main__":
    main()
