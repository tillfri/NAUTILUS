"""
compare_single_class_recall.py — Compare per-class detection metrics across three
prompting strategies: the all-6-class baseline, single-class-only prompting, and
single-class-only prompting with an explicit "return null if absent" instruction.

Loads evaluation_results/evaluation_test_brackish_prompt_with_classes.json (baseline)
plus one evaluation_results/evaluation_brackish_<class>_only.json and one
evaluation_results/evaluation_brackish_<class>_only_null.json per target class, and
prints a table of Precision/Recall/F1/AP@0.5 for that class under all three
strategies, with recall deltas vs. baseline highlighted.

Usage:
    python compare_single_class_recall.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "evaluation_results"
BASELINE_JSON = RESULTS_DIR / "evaluation_test_brackish_prompt_with_classes.json"

TARGET_CLASSES = ["fish", "crab", "jellyfish", "starfish", "shrimp"]


def load_per_class(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)["per_class"]


def main():
    baseline_per_class = load_per_class(BASELINE_JSON)

    rows = []
    for cls in TARGET_CLASSES:
        only_per_class = load_per_class(RESULTS_DIR / f"evaluation_brackish_{cls}_only.json")
        null_per_class = load_per_class(RESULTS_DIR / f"evaluation_brackish_{cls}_only_null.json")

        base = baseline_per_class.get(cls, {})
        only = only_per_class.get(cls, {})
        null = null_per_class.get(cls, {})

        base_recall = base.get("Recall@0.5", 0.0)
        only_recall = only.get("Recall@0.5", 0.0)
        null_recall = null.get("Recall@0.5", 0.0)
        rows.append(
            {
                "class": cls,
                "support": base.get("support", 0),
                "base_recall": base_recall,
                "only_recall": only_recall,
                "null_recall": null_recall,
                "delta_only": only_recall - base_recall,
                "delta_null": null_recall - base_recall,
                "base_precision": base.get("Precision@0.5", 0.0),
                "only_precision": only.get("Precision@0.5", 0.0),
                "null_precision": null.get("Precision@0.5", 0.0),
                "base_f1": base.get("F1@0.5", 0.0),
                "only_f1": only.get("F1@0.5", 0.0),
                "null_f1": null.get("F1@0.5", 0.0),
                "base_num_pred": base.get("num_pred", 0),
                "only_num_pred": only.get("num_pred", 0),
                "null_num_pred": null.get("num_pred", 0),
            }
        )

    header = (
        f"{'class':<12}{'support':>8}"
        f"{'R base':>9}{'R only':>9}{'R null':>9}"
        f"{'P base':>9}{'P only':>9}{'P null':>9}"
        f"{'F1 base':>9}{'F1 only':>9}{'F1 null':>9}"
        f"{'pred base':>11}{'pred only':>11}{'pred null':>11}"
    )
    print("\n" + "=" * len(header))
    print("BASELINE vs. SINGLE-CLASS-ONLY vs. SINGLE-CLASS-ONLY+NULL — per-class comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['class']:<12}{r['support']:>8}"
            f"{r['base_recall']:>9.4f}{r['only_recall']:>9.4f}{r['null_recall']:>9.4f}"
            f"{r['base_precision']:>9.4f}{r['only_precision']:>9.4f}{r['null_precision']:>9.4f}"
            f"{r['base_f1']:>9.4f}{r['only_f1']:>9.4f}{r['null_f1']:>9.4f}"
            f"{r['base_num_pred']:>11}{r['only_num_pred']:>11}{r['null_num_pred']:>11}"
        )
    print("=" * len(header))

    print("\nRecall vs. baseline:")
    for r in rows:
        d_only = "UP" if r["delta_only"] > 1e-9 else ("DOWN" if r["delta_only"] < -1e-9 else "FLAT")
        d_null = "UP" if r["delta_null"] > 1e-9 else ("DOWN" if r["delta_null"] < -1e-9 else "FLAT")
        print(
            f"  {r['class']:<12} only: {d_only:<5} ({r['delta_only']:+.4f})   "
            f"only_null: {d_null:<5} ({r['delta_null']:+.4f})"
        )


if __name__ == "__main__":
    main()
