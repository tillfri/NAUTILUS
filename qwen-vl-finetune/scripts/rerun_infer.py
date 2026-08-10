"""
rerun_infer.py — Stage 1 of the NAUTILUS determinism check: re-run inference on every
image where a previous run made a mistake (per <run-dir>/error_visualizations/errors.json),
sharding the work across one or more GPUs.

Meant to be run inside the `nautilus-qwen` container, where paths recorded in
errors.json/metadata.json (e.g. `/workspace/datasets/...`, `/workspace/weights/...`)
resolve directly.

The combined list of error images across all --run-dir arguments is split into one
shard per requested CUDA device; one worker subprocess per device loads the model
once and runs plain `batch_inference.run_inference` (same code path as a normal
inference run) over its shard. Outputs land at:
    <run-dir>/error_visualizations/rerun_results/<stem>.txt       (raw model text)
    <run-dir>/error_visualizations/rerun_metadata.json            (per-stem input dims)

Run rerun_compare.py afterwards to diff these fresh outputs against the original
predictions recorded in errors.json.

Usage:
    python rerun_infer.py \
        --run-dir /workspace/runs/test_brackish_prompt_with_classes \
        --run-dir /workspace/runs/asterias_rubens_with_class \
        --devices 0,1,2,3 \
        [--limit 50] [--max-new-tokens 2048] [--overwrite]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def build_tasks(run_dirs, limit, overwrite):
    tasks = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        errors_path = run_dir / "error_visualizations" / "errors.json"
        metadata_path = run_dir / "metadata.json"
        with open(errors_path) as f:
            errors = json.load(f)
        with open(metadata_path) as f:
            metadata = json.load(f)

        prompt = metadata["prompt"]
        checkpoint = metadata["checkpoint"]
        out_dir = run_dir / "error_visualizations" / "rerun_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        records = errors["images"]
        if limit:
            records = records[:limit]

        for record in records:
            stem = record["stem"]
            out_path = out_dir / f"{stem}.txt"
            if out_path.exists() and not overwrite:
                continue
            tasks.append(
                {
                    "run_dir": str(run_dir),
                    "stem": stem,
                    "image": record["image"],
                    "checkpoint": checkpoint,
                    "prompt": prompt,
                    "out_path": str(out_path),
                }
            )
    return tasks


def shard(tasks, num_shards):
    shards = [[] for _ in range(num_shards)]
    for i, t in enumerate(tasks):
        shards[i % num_shards].append(t)
    return shards


def run_worker(tasks_file: str, device: str, max_new_tokens: int):
    import torch
    from transformers import AutoProcessor

    sys.path.append(str(SCRIPTS_DIR))
    sys.path.append(str(SCRIPTS_DIR.parent))
    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_VL_Nautilus_ForConditionalGeneration,
    )
    from batch_inference import run_inference

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover

        def tqdm(iterable, **kwargs):
            return iterable

    with open(tasks_file) as f:
        tasks = json.load(f)
    # Group by checkpoint so we only reload the model when it actually changes.
    tasks.sort(key=lambda t: t["checkpoint"])

    dims_path = Path(tasks_file).with_name(f"dims_device{device}.json")
    dims_records = []

    cuda_device = "cuda:" + device
    model = None
    processor = None
    current_checkpoint = None
    min_pixels = 1 * 28 * 28
    max_pixels = 1338 * 28 * 28

    num_ok, num_failed = 0, 0
    for task in tqdm(tasks, desc=f"device {device}"):
        if task["checkpoint"] != current_checkpoint:
            current_checkpoint = task["checkpoint"]
            print(f"[device {device}] loading checkpoint {current_checkpoint}", flush=True)
            model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
                current_checkpoint,
                cache_dir=None,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map=cuda_device,
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(
                current_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )

        image_path = Path(task["image"])
        out_path = Path(task["out_path"])
        try:
            res_text, input_height, input_width = run_inference(
                model, processor, image_path, task["prompt"], max_new_tokens
            )
        except Exception as e:  # noqa: BLE001
            print(f"[device {device}] [FAILED] {task['stem']}: {e}", flush=True)
            num_failed += 1
            continue

        out_path.write_text(res_text)
        dims_records.append(
            {
                "run_dir": task["run_dir"],
                "stem": task["stem"],
                "input_height": input_height,
                "input_width": input_width,
            }
        )
        num_ok += 1
        if num_ok % 20 == 0:
            with open(dims_path, "w") as f:
                json.dump(dims_records, f)

    with open(dims_path, "w") as f:
        json.dump(dims_records, f)

    print(f"[device {device}] done. ok={num_ok} failed={num_failed}", flush=True)


def merge_metadata(run_dirs, log_dir: Path):
    all_dims = []
    for f in sorted(log_dir.glob("dims_device*.json")):
        with open(f) as fh:
            all_dims.extend(json.load(fh))

    by_run = {}
    for d in all_dims:
        by_run.setdefault(d["run_dir"], {})[d["stem"]] = {
            "input_height": d["input_height"],
            "input_width": d["input_width"],
        }

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta_path = run_dir / "error_visualizations" / "rerun_metadata.json"
        existing = {}
        if meta_path.exists():
            with open(meta_path) as f:
                existing = json.load(f)
        existing.update(by_run.get(str(run_dir), {}))
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Wrote {meta_path} ({len(existing)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: parallel multi-GPU rerun of error images."
    )
    parser.add_argument(
        "--run-dir", action="append", help="Run directory (repeatable)."
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated CUDA device ids to shard the work across.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of error images processed per run dir (for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-infer images that already have a rerun result on disk.",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    # Internal worker mode (spawned as a subprocess, one per device) — not for direct use.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tasks-file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.tasks_file, args.device, args.max_new_tokens)
        return

    if not args.run_dir:
        parser.error("--run-dir is required")

    tasks = build_tasks(args.run_dir, args.limit, args.overwrite)
    if not tasks:
        print("Nothing to do — all rerun outputs already exist (use --overwrite to force).")
        return

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    shards = shard(tasks, len(devices))

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path(args.run_dir[0]) / "error_visualizations" / "rerun_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(tasks)} images to rerun, sharded across devices {devices}")

    procs = []
    t0 = time.time()
    for device, shard_tasks in zip(devices, shards):
        if not shard_tasks:
            continue
        tasks_file = log_dir / f"shard_device{device}.json"
        with open(tasks_file, "w") as f:
            json.dump(shard_tasks, f)
        log_file = log_dir / f"device{device}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--tasks-file",
            str(tasks_file),
            "--device",
            device,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        print(f"[device {device}] {len(shard_tasks)} images -> log: {log_file}")
        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((device, p, lf))

    exit_codes = {}
    for device, p, lf in procs:
        p.wait()
        lf.close()
        exit_codes[device] = p.returncode
        print(f"[device {device}] finished with exit code {p.returncode}")

    elapsed = time.time() - t0
    print(f"\nAll shards finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    failed = [d for d, c in exit_codes.items() if c != 0]
    if failed:
        print(f"WARNING: devices with non-zero exit code: {failed} — check logs in {log_dir}")

    merge_metadata(args.run_dir, log_dir)

    print("\nDone. Rerun raw outputs are in <run-dir>/error_visualizations/rerun_results/")
    print("Next: run rerun_compare.py to compare against the original errors.json predictions.")


if __name__ == "__main__":
    main()
