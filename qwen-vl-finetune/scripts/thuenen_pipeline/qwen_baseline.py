"""Stock Qwen2.5-VL-7B-Instruct on Thünen: what is NAUTILUS without the training?

NAUTILUS is Qwen2.5-VL-7B-Instruct plus the VFE branch plus NautData fine-tuning.
``vfe_ablation.py`` showed the **VFE branch** contributes nothing measurable to
detection in the released checkpoint, which leaves one question open: how much of
every NAUTILUS number in this thesis is already in the base model? This script runs
**unmodified** Qwen2.5-VL-7B-Instruct down the same path -- same prompt, same pixel
budget, same decoding, same output layout -- so the three runs

    stock Qwen2.5-VL   <->   NAUTILUS (``vfe_ablation`` condition ``full``)
                       <->   NAUTILUS without VFE (``vfe_identity``)

differ in the checkpoint and nothing else, and go through one scorer.

What is held constant, and how
------------------------------
*Prompt.* ``prompt_variants`` builds the messages, default ``P0_baseline``
(``nautilus_zeroshot.PROMPT_TEMPLATE`` verbatim). No hint of Qwen's own grounding
conventions is added: the question is what the *same* instruction gets out of the two
checkpoints. Qwen's answer is free-form, and often a fenced ``json`` block rather than
NAUTILUS' bare lines -- that is a finding, not a defect, and needs no normalising
because ``visualize_detections.extract_detections_from_text`` (what every scorer here
parses with) pulls ``bbox_2d`` objects out of arbitrary text.

*Pixels.* The same ``--query-max-pixels`` and the same one-step resize as the house
path, so both models see the same grid and write coordinates in the same model-input
pixel space.

*Decoding.* NAUTILUS' ``generation_config.json`` (``do_sample`` with ``top_k: 1``,
``top_p: 0.001``, ``temperature: 0.1``, ``repetition_penalty: 1.05``) is copied onto
the stock model, whose own config is argmax by a different route (``temperature:
1e-6``, no ``top_k``). Both are greedy either way; copying makes it the same sampler
code path, so decoding cannot be offered as an explanation of a difference. The
values applied are recorded in ``metadata.json``.

*Token doubling is NOT held constant, and must not be.* The VFE branch emits two
feature vectors per patch, so the house path duplicates every image placeholder
(``infer_fewshot.double_image_tokens``). Stock Qwen emits one per patch; doubling
there is not a fair-comparison choice but a crash (features/placeholders mismatch).
``run_messages(..., double_tokens=False)`` is the whole difference in the input path,
and ``--self-test`` proves it lands on the right count before any weights are loaded.

Modes
-----
default        Generation over the full test split (``--subsample`` for the pinned
               250-frame screening set), one dir per variant, resumable with
               ``--skip-existing``, shardable with ``--shard i/N``.
``--self-test`` CPU, no weights: renders the P0 messages for one image through the
               real processor and checks the image-placeholder count against the ViT
               token count with and without doubling.
``--merge-metadata``
               Fold ``metadata.shard*.json`` into ``metadata.json``, same convention
               as ``build_proposals.py``.

Usage:
    # 0. the input-shape gate (CPU, seconds)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 qwen_baseline.py --self-test"

    # 1. smoke test: 5 subsample images, dump the rendered prompt
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 qwen_baseline.py --subsample --limit 5 --dump-text --device 0 \\
         --runs /workspace/runs/qwen_baseline_smoke"

    # 2. the full test split, sharded over beta's four GPUs
    for i in 0 1 2 3; do
      docker exec -d nautilus-qwen bash -lc \\
        "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
         python3 qwen_baseline.py --device $i --shard $i/4 --skip-existing"
    done

    # 3. fold the shard metadata once the four are done
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 qwen_baseline.py --merge-metadata"

    # 4. score it exactly like every other run
    docker exec nautilus-qwen python /workspace/NAUTILUS/qwen-vl-finetune/scripts/\
evaluate_detections.py \\
      --gt-dir       /workspace/datasets/thuenen_scaling/test/labels_prompt \\
      --pred-dir     /workspace/runs/qwen_baseline/P0_baseline/results \\
      --image-dir    /workspace/datasets/thuenen_scaling/test/images \\
      --classes-file /workspace/datasets/thuenen_scaling/classes_prompt.txt \\
      --save-json    /workspace/runs/qwen_baseline/P0_baseline/metrics.json

    # 5. adherence table next to the NAUTILUS conditions
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 prompt_report.py --runs /workspace/runs/qwen_baseline"
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

from prompt_experiments import (  # noqa: E402
    DEFAULT_ROOT,
    fingerprint,
    load_subsample,
    query_path_for,
    read_prompt_classes,
)

DEFAULT_CHECKPOINT = "/workspace/weights/qwen2.5-vl-7b-instruct"
DEFAULT_RUNS = "/workspace/runs/qwen_baseline"
DEFAULT_VARIANT = "P0_baseline"

# NAUTILUS' generation_config.json, copied verbatim onto the stock checkpoint. See the
# module docstring: both models decode greedily, this makes it the same code path.
NAUTILUS_GENERATION = {
    "do_sample": True,
    "temperature": 0.1,
    "top_k": 1,
    "top_p": 0.001,
    "repetition_penalty": 1.05,
}


# ── shard / stem plumbing ────────────────────────────────────────────────────
def parse_shard(text):
    """``"i/N"`` -> ``(i, N)``; ``None`` -> ``(0, 1)``. Mirrors build_proposals.py."""
    if not text:
        return 0, 1
    try:
        index, count = text.split("/")
        index, count = int(index), int(count)
    except ValueError:
        raise SystemExit("--shard must be i/N, got {!r}".format(text))
    if count < 1 or not 0 <= index < count:
        raise SystemExit("--shard must be i/N with 0 <= i < N, got {!r}".format(text))
    return index, count


def metadata_name(shard_index, shard_count):
    """One metadata file per shard: concurrent writers would corrupt a shared one."""
    if shard_count == 1:
        return "metadata.json"
    return "metadata.shard{}.json".format(shard_index)


def resolve_stems(args):
    """Full test split by default; the pinned screening subsample on request."""
    if args.subsample:
        stems = list(load_subsample(args.root)["stems"])
    else:
        images_dir = os.path.join(args.root, args.split, "images")
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                       if f.lower().endswith(".jpg"))
    return stems[:args.limit] if args.limit else stems


def merge_metadata(args):
    """Fold every ``metadata.shard*.json`` of a run dir into one ``metadata.json``."""
    for name in variant_names(args):
        out_dir = os.path.join(args.runs, name + args.tag)
        if not os.path.isdir(out_dir):
            print("[skip] {}: no such run dir".format(out_dir))
            continue
        shards = sorted(f for f in os.listdir(out_dir)
                        if f.startswith("metadata.shard") and f.endswith(".json"))
        if not shards:
            print("[skip] {}: no shard metadata to merge".format(out_dir))
            continue
        merged = None
        for shard in shards:
            with open(os.path.join(out_dir, shard), encoding="utf-8") as handle:
                payload = json.load(handle)
            if merged is None:
                merged = payload
                merged.pop("shard", None)
                continue
            merged["image_dims"].update(payload.get("image_dims", {}))
            merged["prompt_tokens"].update(payload.get("prompt_tokens", {}))
            if merged.get("prompt") is None:
                merged["prompt"] = payload.get("prompt")
            if payload.get("gpu_name") not in (None, merged.get("gpu_name")):
                merged["gpu_name"] = "{} + {}".format(merged["gpu_name"],
                                                      payload["gpu_name"])
        # Both are per-shard in the shard files and would otherwise describe shard 0
        # alone; the merged run is the union, and its fingerprint is what a later
        # comparison has to match against another run's.
        merged["subsample_n"] = len(merged["image_dims"])
        merged["subsample_sha1"] = fingerprint(
            os.path.splitext(name)[0] for name in merged["image_dims"])
        merged["shard"] = "merged {} shards".format(len(shards))
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2)
        n_results = len([f for f in os.listdir(os.path.join(out_dir, "results"))
                         if f.endswith(".txt")])
        flag = "" if n_results == len(merged["image_dims"]) else \
            "   <-- MISMATCH, a shard is incomplete"
        print("{}: {} shards -> {} image_dims against {} result files{}".format(
            name, len(shards), len(merged["image_dims"]), n_results, flag))
    return 0


def variant_names(args):
    return [v.strip() for v in args.variants.split(",") if v.strip()]


# ── --self-test ──────────────────────────────────────────────────────────────
def self_test(args):
    """Prove the placeholder count, before any weights are loaded.

    The single input-path difference from the NAUTILUS runs is that the image
    placeholders are *not* duplicated. Qwen2.5-VL feeds the LLM one merged token per
    2x2 patch block, i.e. ``t*h*w / merge_size**2`` per image; the house path doubles
    that for the VFE's second feature vector. Getting this wrong is not a silent
    failure -- transformers raises on the mismatch -- but the gate is free and states
    the invariant where it can be read.
    """
    from transformers import AutoProcessor

    from batch_inference import image_token_id
    from prompt_experiments import build_inputs
    from prompt_variants import Context, VARIANTS
    from infer_fewshot import double_image_tokens

    name = variant_names(args)[0]
    stems = list(load_subsample(args.root)["stems"])
    images_dir = os.path.join(args.root, args.split, "images")
    query_path = query_path_for(images_dir, stems[0])

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=1 * 28 * 28,
        max_pixels=args.query_max_pixels * 28 * 28)
    ctx = Context(read_prompt_classes(args.root),
                  args.query_max_pixels * 28 * 28,
                  args.exemplar_max_pixels * 28 * 28)
    messages = VARIANTS[name]["build"](ctx, query_path)
    _text, inputs = build_inputs(processor, messages)

    grid = inputs["image_grid_thw"][0]
    merge = processor.image_processor.merge_size
    expected = int(grid[0] * grid[1] * grid[2]) // (merge * merge)
    plain = int((inputs["input_ids"][0] == image_token_id).sum())
    doubled_ids, _mask = double_image_tokens(inputs, image_token_id)
    doubled = int((doubled_ids[0] == image_token_id).sum())

    print("{}  grid {}x{}x{}  merge {}".format(
        stems[0], int(grid[0]), int(grid[1]), int(grid[2]), merge))
    print("  ViT tokens the LLM is handed (stock)   {}".format(expected))
    print("  image placeholders, no doubling        {}".format(plain))
    print("  image placeholders, house doubling     {}".format(doubled))
    failures = []
    if plain != expected:
        failures.append("undoubled placeholders {} != ViT tokens {}".format(
            plain, expected))
    if doubled != 2 * expected:
        failures.append("doubled placeholders {} != 2x ViT tokens {}".format(
            doubled, 2 * expected))
    if failures:
        for failure in failures:
            print("  [FAIL] {}".format(failure))
        return 1
    print("  [ok] stock Qwen needs double_tokens=False; NAUTILUS needs True")
    return 0


# ── inference ────────────────────────────────────────────────────────────────
def load_model(args):
    """Stock Qwen2.5-VL, loaded exactly as the house path loads NAUTILUS."""
    sys.path.insert(0, os.path.dirname(SCRIPTS))
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint, cache_dir=None, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, device_map="cuda:" + str(args.device))
    model.eval()
    applied = {}
    for key, value in NAUTILUS_GENERATION.items():
        applied[key] = value
        setattr(model.generation_config, key, value)
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=1 * 28 * 28,
        max_pixels=args.query_max_pixels * 28 * 28)
    return model, processor, applied


def load_exemplars(args, spec):
    if not spec["exemplars"]:
        return []
    path = os.path.join(args.root, "exemplars_k{}.json".format(spec["exemplars"]))
    if not os.path.isfile(path):
        raise SystemExit("{} not found -- run prompt_experiments.py --make-exemplars "
                         "first".format(path))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_variant(args, model, processor, generation, name, stems, prompt_names,
                images_dir, shard):
    """Mirror of ``vfe_ablation.run_condition`` for the stock checkpoint."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    from nautilus_zeroshot import snap_labels
    from prompt_experiments import annotator, run_messages
    from prompt_variants import Context, VARIANTS

    shard_index, shard_count = shard
    spec = VARIANTS[name]
    out_dir = os.path.join(args.runs, name + args.tag)
    results_dir = os.path.join(out_dir, "results")
    raw_dir = os.path.join(out_dir, "results_raw")
    for directory in (results_dir, raw_dir):
        os.makedirs(directory, exist_ok=True)
    metadata_path = os.path.join(out_dir, metadata_name(shard_index, shard_count))

    gpu_name = torch.cuda.get_device_name(model.device)
    previous = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    if previous.get("gpu_name") and previous["gpu_name"] != gpu_name:
        raise SystemExit(
            "{} was started on {} and would resume on {}. Greedy outputs differ across "
            "GPU generations, so a mixed run is not one run.".format(
                out_dir, previous["gpu_name"], gpu_name))

    exemplars = load_exemplars(args, spec)
    ctx = Context(prompt_names, args.query_max_pixels * 28 * 28,
                  args.exemplar_max_pixels * 28 * 28,
                  annotate=annotator(os.path.join(args.runs, "_annotated_exemplars"))
                  if spec.get("annotated") else None)

    metadata = {
        "prompt": previous.get("prompt"),
        "checkpoint": args.checkpoint,
        "image_dims": previous.get("image_dims", {}),
        "variant": name,
        "note": spec.get("note"),
        "subsample_sha1": fingerprint(stems),
        "subsample_n": len(stems),
        "exemplars": [e["stem"] for e in exemplars],
        "exemplar_max_pixels": args.exemplar_max_pixels,
        "query_max_pixels": args.query_max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "snapped": spec.get("snap", True),
        "prompt_tokens": previous.get("prompt_tokens", {}),
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "double_tokens": False,
        "generation": generation,
        "split": "subsample" if args.subsample else args.split,
        "shard": "{}/{}".format(shard_index, shard_count),
        "gpu_name": gpu_name,
        "device": "cuda:{}".format(args.device),
    }
    stats = Counter()
    over_budget = []
    description = name if shard_count == 1 else "{} [{}/{}]".format(
        name, shard_index, shard_count)

    try:
        for stem in tqdm(stems, desc=description):
            raw_path = os.path.join(raw_dir, stem + ".txt")
            out_path = os.path.join(results_dir, stem + ".txt")
            query_path = query_path_for(images_dir, stem)
            if args.skip_existing and os.path.isfile(raw_path) and os.path.isfile(out_path):
                stats["skipped"] += 1
                if os.path.basename(query_path) not in metadata["image_dims"]:
                    print("[warning] {} was already written but has no metadata entry; "
                          "delete it to regenerate".format(stem))
                continue
            try:
                response, input_height, input_width, prompt_tokens, text = run_messages(
                    model, processor, spec, ctx, exemplars, query_path,
                    args.max_new_tokens, double_tokens=False)
            except Exception as error:  # noqa: BLE001
                print("[failed] {}: {}".format(stem, error))
                stats["failed"] += 1
                continue

            if metadata["prompt"] is None:
                metadata["prompt"] = text
                if args.dump_text:
                    dump = os.path.join(out_dir, "rendered_prompt.txt")
                    with open(dump, "w", encoding="utf-8") as handle:
                        handle.write(text)
                    print("\nrendered prompt for {} -> {} ({} tokens)".format(
                        stem, dump, prompt_tokens))
            if prompt_tokens > args.token_warn:
                over_budget.append((stem, prompt_tokens))

            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write(response)
            if spec.get("snap", True):
                snapped, unmatched = snap_labels(response, prompt_names)
                stats["unmatched_labels"] += unmatched
            else:
                snapped = response
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(snapped)

            with Image.open(query_path) as image:
                width, height = image.size
            metadata["image_dims"][os.path.basename(query_path)] = {
                "input_height": input_height, "input_width": input_width,
                "original_width": width, "original_height": height,
            }
            metadata["prompt_tokens"][stem] = prompt_tokens
            stats["ok"] += 1
            if stats["ok"] % 50 == 0:
                with open(metadata_path, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)
    finally:
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print("{}: ok={} skipped={} failed={} unmatched_labels={} on {}".format(
        description, stats["ok"], stats["skipped"], stats["failed"],
        stats["unmatched_labels"], gpu_name))
    tokens = list(metadata["prompt_tokens"].values())
    if tokens:
        print("  prompt_tokens mean {:.0f} max {}".format(
            sum(tokens) / len(tokens), max(tokens)))
    if over_budget:
        print("  [warning] {} images over {} prompt tokens (max {})".format(
            len(over_budget), args.token_warn, max(t for _, t in over_budget)))
    return stats


def run_inference(args):
    from prompt_variants import VARIANTS

    requested = variant_names(args)
    unknown = [v for v in requested if v not in VARIANTS]
    if unknown:
        raise SystemExit("unknown variant(s): {}\nknown: {}".format(
            ", ".join(unknown), ", ".join(sorted(VARIANTS))))

    shard = parse_shard(args.shard)
    stems = resolve_stems(args)
    if shard[1] > 1:
        stems = stems[shard[0]::shard[1]]
        print("shard {} of {}".format(shard[0], shard[1]))
    prompt_names = read_prompt_classes(args.root)
    images_dir = os.path.join(args.root, args.split, "images")
    print("{} variants x {} images on cuda:{} ({})".format(
        len(requested), len(stems), args.device, args.checkpoint))

    model, processor, generation = load_model(args)
    for name in requested:
        run_variant(args, model, processor, generation, name, stems, prompt_names,
                    images_dir, shard)
    if shard[1] > 1:
        print("\nshard done; fold the metadata once every shard has finished with:\n"
              "  python3 qwen_baseline.py --merge-metadata --runs {} --variants {}"
              .format(args.runs, args.variants))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="The thuenen_scaling split root.")
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Stock Qwen2.5-VL-7B-Instruct.")
    parser.add_argument("--variants", default=DEFAULT_VARIANT,
                        help="Comma-separated prompt_variants names.")
    parser.add_argument("--subsample", action="store_true",
                        help="The pinned 250-frame screening set instead of the split.")
    parser.add_argument("--shard", default=None, metavar="i/N",
                        help="Run stems [i::N] and write metadata.shard<i>.json.")
    parser.add_argument("--merge-metadata", action="store_true",
                        help="Fold metadata.shard*.json into metadata.json and exit.")
    parser.add_argument("--self-test", action="store_true",
                        help="CPU input-shape gate; no weights are loaded.")
    parser.add_argument("--device", default="0", help="CUDA device index.")
    parser.add_argument("--tag", default="", help="Suffix for the run dir name.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--query-max-pixels", type=int, default=1338,
                        help="Query budget in 28x28 patches; the house value.")
    parser.add_argument("--exemplar-max-pixels", type=int, default=256,
                        help="Per-exemplar budget in 28x28 patches.")
    parser.add_argument("--token-warn", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None,
                        help="First N stems, for a smoke test.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume: skip stems that already have both output files.")
    parser.add_argument("--dump-text", action="store_true",
                        help="Write the first rendered prompt to rendered_prompt.txt.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge_metadata:
        return merge_metadata(args)
    if args.self_test:
        return self_test(args)
    return run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
