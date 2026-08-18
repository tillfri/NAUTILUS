"""Re-run a finished ``yolo_init_comparison.py`` run's prediction pass and diff the
box counts against the ones recorded in its ``result.json``.

The sweep materialises its whole confidence sweep from a single GPU pass, so the
only thing worth re-checking is that pass. This loads ``best.pt`` fresh, runs
``predict_split`` at the lowest threshold, re-filters at every threshold in the
recorded ``conf_sweep``, and prints a per-threshold diff.

Exact agreement is not expected. The predictions were made while four trainings
shared the box, and cuDNN picks kernels by what is free; the resulting float
non-associativity moves a handful of boxes across the lowest thresholds. The
check is that the drift is small and shrinks as the threshold rises -- a
sign-flip-sized disagreement would mean the wrong checkpoint.

Usage:
  docker exec brackish python \
    /usr/src/ultralytics/brackish/determinism_check.py \
    --run /usr/src/ultralytics/brackish/thuenen_scaling_x1280_runs/mega_n8_s0 \
    --device 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_scaling import predict_split          # noqa: E402
from ultralytics import YOLO                    # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True,
                        help="run directory holding result.json")
    parser.add_argument("--root", default="/usr/src/ultralytics/brackish/thuenen_scaling",
                        help="dataset root whose test split was predicted")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    with open(os.path.join(args.run, "result.json")) as handle:
        record = json.load(handle)

    model = YOLO(record["weights"], task="detect")
    collected = predict_split(model, args.root, "test", record["conf_sweep"][0]["conf"],
                              args.device, imgsz=args.imgsz)

    print(f"run     {os.path.basename(args.run.rstrip('/'))}")
    print(f"weights {record['weights']}")
    print(f"images  {len(collected)}")
    print(f"{'tag':>5} {'rerun':>8} {'recorded':>9} {'delta':>7} {'drift':>8}")

    worst = 0.0
    for entry in record["conf_sweep"]:
        rerun = sum(1 for _, _, _, boxes in collected
                    for _, _, score in boxes if score >= entry["conf"])
        delta = rerun - entry["boxes"]
        drift = abs(delta) / entry["boxes"] if entry["boxes"] else 0.0
        worst = max(worst, drift)
        print(f"{entry['tag']:>5} {rerun:8d} {entry['boxes']:9d} {delta:+7d} {drift:7.2%}")

    print(f"worst drift {worst:.2%}")


if __name__ == "__main__":
    main()
