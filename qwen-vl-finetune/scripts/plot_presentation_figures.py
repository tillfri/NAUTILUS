"""Presentation figures for the Thünen zero-shot vs. supervised comparison.

Renders the four plots the intermediate-presentation deck
(``demonstration/presentation_week2.md``) is built around. Every number comes from
the ``localization.json`` files ``localization_report.py`` already wrote — no
inference and no re-scoring happens here, so a figure can never drift from the
published tables in ``masterarbeit/thuenen-scaling-experiment.md`` and
``masterarbeit/megalodon-classagnostic.md``.

    budget_curve.png     the hero plot: the same models and frames scored two ways.
                         At IoU 0.5 YOLOv8m wins at every budget; at centre-in-box
                         zero-shot NAUTILUS ties YOLOv8m trained on 32 images/class.
    iou_sweep.png        recall vs. IoU threshold. NAUTILUS's recall triples between
                         IoU 0.5 and 0.2 — the signature of a box-scale mismatch on
                         animals that *were* found, not of hallucination.
    convention_ratio.png recall on circle-derived (square) GT / recall on
                         rectangle-drawn GT. Separates convention-naive models
                         (< 1) from models that learned the annotation UI (> 1).
    box_convention.png   why: predicted box area vs. GT, and the share of exactly
                         square boxes climbing toward the GT's 24.9%.

This runs on the **host** interpreter — it needs only matplotlib and the json
files, no torch. It is the one script here that does not need the container.

Usage:
    python3 /home/tfricke/nautilus/NAUTILUS/qwen-vl-finetune/scripts/plot_presentation_figures.py

    # or, if the host matplotlib is ever unavailable (mounts mirror, paths resolve):
    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts && python3 plot_presentation_figures.py \
         --runs-dir /workspace/runs --out-dir /workspace/runs/../demonstration/figures"
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- palette -----------------------------------------------------------------
# Validated (dataviz skill, light surface, all-pairs): worst CVD dE 9.2, worst
# normal-vision dE 24.0. Colour follows the *entity*, identically in all four
# figures, so "orange is NAUTILUS" holds across the whole deck.
YOLO = "#2a78d6"  # slot 1, blue    — supervised YOLOv8m
NAUT = "#eb6834"  # slot 2, orange  — NAUTILUS zero-shot
MEGA = "#1baf7a"  # slot 3, aqua    — frozen FathomNet Megalodon
# MEGA sits at 2.74:1 on the light surface, below the 3:1 bar, so every figure
# that uses it carries visible direct labels (the skill's relief rule).

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BUDGETS = ["1", "2", "4", "8", "16", "32", "64", "128", "full"]
TRAIN_IMAGES = {  # realised training images per budget, for the axis subtitle
    "1": 28, "2": 52, "4": 108, "8": 211, "16": 404,
    "32": 714, "64": 1221, "128": 2003, "full": 4275,
}
MEGA_CONFS = ["001", "005", "010", "015", "025", "040", "050", "060"]
MEGA_MATCHED = "050"  # the threshold where Megalodon emits NAUTILUS's box budget

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


# --- data --------------------------------------------------------------------

def load_runs(runs_dir: Path) -> dict:
    """Read every localization.json this deck needs, keyed by a short name."""
    wanted = {"nautilus": "thuenen_zeroshot"}
    wanted.update({f"yolo_{b}": f"thuenen_yolo_n{b}_s0" for b in BUDGETS})
    wanted.update({f"mega_{c}": f"thuenen_megalodon_i1280_c{c}" for c in MEGA_CONFS})

    runs = {}
    for key, dirname in wanted.items():
        path = runs_dir / dirname / "localization.json"
        if not path.exists():
            raise SystemExit(f"missing {path} — run localization_report.py for {dirname} first")
        runs[key] = json.loads(path.read_text())
    return runs


def recall(run: dict, criterion: str) -> float:
    return run["criteria"][criterion]["recall"]


def convention_ratio(run: dict, criterion: str = "iou_0.5") -> float:
    """Recall on circle-derived (square) GT divided by recall on rectangle GT."""
    c = run["criteria"][criterion]
    return c["square_gt"]["recall"] / c["rect_gt"]["recall"]


def style_axes(ax, ygrid=True, xgrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x", visible=ygrid or xgrid, linestyle="-")
    if xgrid:
        ax.grid(axis="x", linestyle="-")
    if not ygrid:
        ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --- plot 1: the hero ---------------------------------------------------------

def plot_budget_curve(runs: dict, out: Path) -> None:
    """The same models, frames and code — only the matching criterion changes."""
    x = range(len(BUDGETS))
    # note_xy / cross_off are hand-placed per panel: the free space is in
    # opposite corners because the two curves cross the NAUTILUS line in
    # different places.
    panels = [
        ("iou_0.5", "Reproducing this dataset\nclass-agnostic recall @ IoU $\\geq$ 0.5",
         "the criterion the circle-tool\nartefact acts on", (0.03, 0.97), "left", "top",
         (8, -38), "left"),
        ("centre_in_box", "Finding the animal\nclass-agnostic recall, centre-in-box",
         "scale-free criterion:\nthe artefact cannot act on it", (0.97, 0.03), "right",
         "bottom", (-10, 22), "right"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)

    for ax, (criterion, title, note, note_xy, note_ha, note_va,
             cross_off, cross_ha) in zip(axes, panels):
        yolo = [recall(runs[f"yolo_{b}"], criterion) for b in BUDGETS]
        naut = recall(runs["nautilus"], criterion)

        ax.axhline(naut, color=NAUT, linewidth=2, linestyle="--", zorder=2)
        ax.plot(x, yolo, color=YOLO, linewidth=2, marker="o", markersize=8,
                markerfacecolor=YOLO, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)

        # Crossing point. TIE_TOL is one unit in the third decimal: at
        # centre-in-box n=32 reads 0.3766 against NAUTILUS's 0.3772, which the
        # write-ups quote as an exact tie at 0.377. A strict >= would skip it and
        # report n=64, overstating the supervised model by a whole budget step.
        TIE_TOL = 0.002
        crossing = next((i for i, v in enumerate(yolo) if v >= naut - TIE_TOL), None)
        if crossing is not None:
            verb = "ties at" if abs(yolo[crossing] - naut) <= TIE_TOL else "crosses at"
            ax.axvline(crossing, color=MUTED, linewidth=0.8, zorder=1)
            ax.annotate(
                f"{verb} n={BUDGETS[crossing]}\n({TRAIN_IMAGES[BUDGETS[crossing]]} images)",
                xy=(crossing, yolo[crossing]), xytext=cross_off, textcoords="offset points",
                fontsize=12, color=INK, ha=cross_ha, linespacing=1.4,
            )

        # direct labels (both series named on the plot, not by colour alone)
        ax.annotate("NAUTILUS zero-shot\n0 training images", xy=(0, naut),
                    xytext=(2, 8), textcoords="offset points",
                    fontsize=12, color=NAUT, fontweight="bold", va="bottom")
        ax.annotate("YOLOv8m", xy=(len(BUDGETS) - 1, yolo[-1]),
                    xytext=(-4, 10), textcoords="offset points",
                    fontsize=12, color=YOLO, fontweight="bold", ha="right")

        ax.set_title(title, color=INK, pad=14, linespacing=1.4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("training images per class")
        ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
        ax.set_ylim(0, 0.72)
        style_axes(ax)
        ax.text(*note_xy, note, transform=ax.transAxes, fontsize=12,
                color=INK_2, ha=note_ha, va=note_va, linespacing=1.4)

    axes[0].set_ylabel("recall on all 1564 test boxes")
    fig.legend(
        handles=[
            Line2D([], [], color=YOLO, lw=2, marker="o", markersize=8,
                   markeredgecolor=SURFACE, markeredgewidth=2, label="YOLOv8m (supervised)"),
            Line2D([], [], color=NAUT, lw=2, ls="--", label="NAUTILUS (zero-shot)"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("Same models, same frames, same code — only the matching criterion changes:\n"
                 "zero-shot NAUTILUS is worth 108 annotated images, or 714, depending on how it is scored",
                 fontsize=15, color=INK, y=0.995, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --- plot 2: the IoU sweep ----------------------------------------------------

def plot_iou_sweep(runs: dict, out: Path) -> None:
    """Relaxing IoU triples NAUTILUS's recall; YOLO's curve is comparatively flat."""
    thresholds = ["iou_0.5", "iou_0.4", "iou_0.3", "iou_0.2", "iou_0.1"]
    labels = ["0.5", "0.4", "0.3", "0.2", "0.1"]
    # (legend name, run key, colour, short direct label, label y-nudge in points)
    # NAUTILUS and Megalodon land within 0.012 of each other at centre-in-box —
    # that near-tie is the finding, but the two labels would sit on top of each
    # other, so they are nudged apart.
    series = [
        ("NAUTILUS zero-shot", "nautilus", NAUT, "NAUTILUS", 11),
        (f"Megalodon (frozen, conf 0.{MEGA_MATCHED[1:]})", f"mega_{MEGA_MATCHED}",
         MEGA, "Megalodon", -12),
        ("YOLOv8m, full train split", "yolo_full", YOLO, "YOLOv8m full", 0),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = list(range(len(thresholds)))
    x_centre = len(thresholds) + 0.6

    for name, key, colour, short, dy in series:
        ys = [recall(runs[key], t) for t in thresholds]
        ax.plot(x, ys, color=colour, linewidth=2, marker="o", markersize=8,
                markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=2)
        centre = recall(runs[key], "centre_in_box")
        ax.plot([x_centre], [centre], color=colour, marker="D", markersize=9,
                markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.plot([x[-1], x_centre], [ys[-1], centre], color=colour, linewidth=1,
                linestyle=":", alpha=0.7)
        # direct label — required, MEGA is below the 3:1 contrast bar
        ax.annotate(short, xy=(x_centre, centre), xytext=(10, dy),
                    textcoords="offset points", fontsize=12, color=colour,
                    fontweight="bold", va="center")

    naut = runs["nautilus"]
    ax.annotate(
        f"{recall(naut, 'iou_0.5'):.1%} → {recall(naut, 'iou_0.2'):.1%}\nrecall triples",
        xy=(3, recall(naut, "iou_0.2")), xytext=(-6, 12), textcoords="offset points",
        fontsize=12, color=NAUT, ha="right", linespacing=1.4,
    )

    ax.axvline(len(thresholds) - 0.3, color=AXIS, linewidth=0.8)
    ax.set_xticks(x + [x_centre])
    ax.set_xticklabels(labels + ["centre\nin box"])
    ax.set_xlabel("IoU threshold (relaxing →)")
    ax.set_ylabel("class-agnostic recall on 1564 test boxes")
    ax.set_xlim(-0.3, x_centre + 2.4)
    ax.set_ylim(0, 0.72)
    ax.set_title("A box-convention mismatch, not hallucination:\n"
                 "NAUTILUS gains most from relaxing the threshold",
                 color=INK, pad=14, linespacing=1.4)
    style_axes(ax)
    ax.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", markersize=8,
                              markeredgecolor=SURFACE, markeredgewidth=2, label=n)
                       for n, _, c, _s, _d in series],
              loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --- plot 3: the convention ratio --------------------------------------------

def plot_convention_ratio(runs: dict, out: Path) -> None:
    """Every model that never saw these labels is penalised on circle-derived GT."""
    rows = [(f"Megalodon conf 0.{c[1:]}", f"mega_{c}", MEGA) for c in MEGA_CONFS]
    rows += [("NAUTILUS zero-shot", "nautilus", NAUT)]
    rows += [(f"YOLOv8m n={b}" if b != "full" else "YOLOv8m full",
              f"yolo_{b}", YOLO) for b in BUDGETS]

    names = [r[0] for r in rows]
    ratios = [convention_ratio(runs[r[1]]) for r in rows]
    colours = [r[2] for r in rows]
    y = list(range(len(rows)))[::-1]

    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    ax.axvspan(0, 1, color=GRID, alpha=0.45, zorder=0)
    ax.barh(y, ratios, height=0.55, color=colours, zorder=3)
    ax.axvline(1.0, color=INK_2, linewidth=1.2, zorder=4)

    for yi, r in zip(y, ratios):
        ax.annotate(f"{r:.2f}", xy=(r, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12, color=INK_2)
    ax.set_xlabel("recall on circle-derived (square) GT  ÷  recall on rectangle-drawn GT   (IoU $\\geq$ 0.5)")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.8, len(rows) + 1.2)  # headroom for the two region captions
    ax.set_title("Only the models trained on these labels are paid for reproducing them",
                 color=INK, pad=16)
    style_axes(ax, ygrid=False, xgrid=True)
    ax.tick_params(axis="y", length=0)

    for xpos, caption in ((0.5, "convention-naive\npenalised on circle GT"),
                          (1.52, "learned the convention\nrewarded by IoU")):
        ax.text(xpos, len(rows) + 1.05, caption, fontsize=12, color=INK_2,
                ha="center", va="top", linespacing=1.4)

    fig.legend(handles=[Patch(facecolor=MEGA, label="Megalodon (frozen, never saw Thünen)"),
                        Patch(facecolor=NAUT, label="NAUTILUS (zero-shot)"),
                        Patch(facecolor=YOLO, label="YOLOv8m (trained on this GT)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --- plot 4: the mechanism ----------------------------------------------------

def plot_box_convention(runs: dict, out: Path) -> None:
    """Two panels, never two y-scales: box scale and the square-box share."""
    rows = [("NAUTILUS\n0-shot", "nautilus", NAUT),
            (f"Megalodon\nconf 0.{MEGA_MATCHED[1:]}", f"mega_{MEGA_MATCHED}", MEGA)]
    rows += [(f"YOLO\nn={b}" if b != "full" else "YOLO\nfull", f"yolo_{b}", YOLO)
             for b in BUDGETS]

    names = [r[0] for r in rows]
    colours = [r[2] for r in rows]
    x = list(range(len(rows)))

    gt = runs["nautilus"]["gt_boxes"]
    gt_area = gt["median_area_frac"]
    gt_square = gt["square_frac"] * 100

    areas = [runs[r[1]]["pred_boxes"]["median_area_frac"] / gt_area for r in rows]
    squares = [runs[r[1]]["pred_boxes"]["square_frac"] * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax1.bar(x, areas, width=0.5, color=colours)
    ax1.axhline(1.0, color=INK_2, linewidth=1.2)
    ax1.text(-0.45, 1.70, "reference line: ground truth = 1.00×",
             fontsize=11, color=INK_2, ha="left", va="center")
    for xi, v in zip(x, areas):
        ax1.annotate(f"{v:.2f}×", xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=11, color=INK_2)
    ax1.set_ylabel("median box area\n÷ ground-truth median")
    ax1.set_ylim(0, 1.8)
    ax1.set_title("NAUTILUS draws boxes half the annotated size", color=INK, pad=10)
    style_axes(ax1)

    ax2.bar(x, squares, width=0.5, color=colours)
    ax2.axhline(gt_square, color=INK_2, linewidth=1.2)
    ax2.annotate(f"ground truth = {gt_square:.1f}%", xy=(len(rows) - 0.5, gt_square),
                 xytext=(0, 6), textcoords="offset points", fontsize=11,
                 color=INK_2, ha="right")
    for xi, v in zip(x, squares):
        ax2.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=11, color=INK_2)
    ax2.set_ylabel("predictions that are\nexactly square (%)")
    ax2.set_ylim(0, 30)
    ax2.set_title("…and only the supervised model learns to emit the circle tool's squares",
                  color=INK, pad=10)
    style_axes(ax2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11)
    ax2.tick_params(axis="x", length=0)

    fig.legend(handles=[Patch(facecolor=NAUT, label="NAUTILUS (zero-shot)"),
                        Patch(facecolor=MEGA, label="Megalodon (frozen)"),
                        Patch(facecolor=YOLO, label="YOLOv8m (supervised)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --- verification printout ----------------------------------------------------

def print_checks(runs: dict) -> None:
    """Print the values the deck quotes, so a figure can be checked against the notes."""
    gt = runs["nautilus"]["gt_boxes"]
    print("\nground truth  "
          f"{gt['boxes']} boxes / {gt['boxes_per_image']:.2f} per image / "
          f"{gt['median_area_frac']:.5f} median area / {gt['square_frac']:.4f} square")

    naut = runs["nautilus"]
    print(f"NAUTILUS      IoU0.5 {recall(naut, 'iou_0.5'):.4f}  "
          f"IoU0.3 {recall(naut, 'iou_0.3'):.4f}  "
          f"centre {recall(naut, 'centre_in_box'):.4f}  "
          f"ratio {convention_ratio(naut):.2f}  "
          f"{naut['pred_boxes']['boxes']} boxes")

    for b in BUDGETS:
        r = runs[f"yolo_{b}"]
        print(f"YOLO n={b:<4}    IoU0.5 {recall(r, 'iou_0.5'):.4f}  "
              f"IoU0.3 {recall(r, 'iou_0.3'):.4f}  "
              f"centre {recall(r, 'centre_in_box'):.4f}  "
              f"ratio {convention_ratio(r):.2f}  "
              f"empty {r['pred_boxes']['empty_image_frac']:.1%}")

    m = runs[f"mega_{MEGA_MATCHED}"]
    print(f"Megalodon@.{MEGA_MATCHED[1:]} IoU0.5 {recall(m, 'iou_0.5'):.4f}  "
          f"IoU0.3 {recall(m, 'iou_0.3'):.4f}  "
          f"centre {recall(m, 'centre_in_box'):.4f}  "
          f"ratio {convention_ratio(m):.2f}")
    m01 = runs["mega_001"]
    print(f"Megalodon@.01 centre {recall(m01, 'centre_in_box'):.4f}  "
          f"IoU0.3 {recall(m01, 'iou_0.3'):.4f}  "
          f"{m01['pred_boxes']['boxes_per_image']:.2f} boxes/img")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("/home/tfricke/nautilus/runs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/tfricke/nautilus/demonstration/figures"))
    args = ap.parse_args()

    runs = load_runs(args.runs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in [("budget_curve.png", plot_budget_curve),
                     ("iou_sweep.png", plot_iou_sweep),
                     ("convention_ratio.png", plot_convention_ratio),
                     ("box_convention.png", plot_box_convention)]:
        path = args.out_dir / name
        fn(runs, path)
        print(f"wrote {path}")

    print_checks(runs)


if __name__ == "__main__":
    main()
