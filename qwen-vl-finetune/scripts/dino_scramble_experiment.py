"""What NAUTILUS' patch scramble does to DINOv2's *output*.

``visualize_dino_input.py`` shows the mangled image the VFE hands its frozen DINOv2
(see ``masterarbeit/nautilus-model-architecture.md``, "Two things that look like
bugs"). This script measures what that costs downstream: the same frozen ViT-L/14 is
run twice per frame --

  B ("correct")    the smart_resize'd frame in true raster layout,
  A ("as shipped") the same frame after ``scramble_patches`` -- i.e. exactly what
                   ``restore_image_from_patches`` builds out of Qwen's
                   2x2-merge-block patch sequence,

-- and the two token sets are compared patch-for-patch on *true* geometry. Because
DINOv2 is a ViT/14 and the reassembled image is exactly ``gh*14 x gw*14``, the patch
grids coincide: the scramble is a pure permutation of intact patches, so token *n*
carries the right pixels at the wrong coordinates and every difference between A and B
comes from the positional embedding.

Both conditions get Qwen's CLIP normalisation (defect #2 held fixed), so the
permutation is the only variable.

Three figures per rendered frame:

  pca.png     joint-PCA-to-RGB feature maps, B next to A, shared colour space
  cosine.png  per-patch cos(A_aligned, B) + histogram
  gain.png    RMS w1 per patch for B and A, plus the relative change of the applied
              gain -- w1 feeds ``weight_2 = 1/exp(-w1)``, the per-token gain the VFE
              multiplies onto Qwen's visual tokens, so this is the quantity that
              actually reaches the LLM (the plain channel mean cancels itself out)

DINOv2 is loaded standalone from ``weights/dinov2-weights/dinov2.pth`` (verified
bit-identical to the ``visual.nautilus_encoder.*`` tensors in the 7B checkpoint cast to
bf16 -- the encoder was frozen for all of NAUTILUS training), and ``nautilus_w1_mlp`` is
read tensor-by-tensor out of the checkpoint shards. The 17 GB LLM is never built.

Usage (container, one GPU):

    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts && \
       python3 dino_scramble_experiment.py \
         --out /workspace/runs/dino_scramble"

    # sanity gates, run these first
    docker exec nautilus-qwen bash -lc \
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts && \
       python3 dino_scramble_experiment.py --identity-check --noise-floor --limit 4 \
         --out /workspace/runs/dino_scramble"

Resumable: a stem already present in ``metrics.csv`` is not recomputed, though it is
still rendered if it has no figures yet and ``--render`` budget is left.

Note on the grid: ``smart_resize`` on the original frame gives 54x96 for a 2704x1520
Thuenen frame, which is the single-resize path (what ``prompt_experiments.py`` feeds).
``batch_inference.py`` omits ``max_pixels`` in the message dict and so resamples twice,
landing on 54x98. Both grids are even and scramble the same way; the numbers here are
not sensitive to the difference.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from visualize_dino_input import (  # noqa: E402
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    PATCH_SIZE,
    QWEN_MEAN,
    QWEN_STD,
    scramble_patches,
    smart_resize,
)

DEFAULT_DATASET = "/workspace/datasets/thuenen_scaling"
DEFAULT_DINO_WEIGHTS = "/workspace/weights/dinov2-weights/dinov2.pth"
DEFAULT_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"
DINO_LAYER = 23  # the block ehance_embeds asks for

CSV_FIELDS = [
    "stem", "grid_h", "grid_w", "n_patches",
    "cos_mean", "cos_median", "cos_p05", "cos_frac_below_0.5",
    "w1_rms_b", "w1_rms_a", "w1_abs_delta_mean",
    "gain_p01_b", "gain_p99_b", "gain_p01_a", "gain_p99_a",
    "gain_reldiff_mean", "gain_reldiff_median",
]


# --------------------------------------------------------------------------- data


def load_frame(image_path, min_pixels, max_pixels):
    """Return (resized_rgb_01, clip_normalised) -- the house preprocessing path."""
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    h, w = smart_resize(h0, w0, min_pixels=min_pixels, max_pixels=max_pixels)
    resized = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float64) / 255.0
    return resized, (resized - QWEN_MEAN) / QWEN_STD


def build_src_permutation(grid_h, grid_w):
    """``src[k]`` = the true raster patch index sitting in fed raster cell ``k``.

    Derived by pushing an index image through ``scramble_patches`` itself rather than
    by re-deriving the index algebra, so it cannot drift from the function the model's
    layout is actually reproduced by.
    """
    ids = np.arange(grid_h * grid_w, dtype=np.float64).reshape(grid_h, grid_w)
    img = np.repeat(np.repeat(ids, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    img = np.repeat(img[..., None], 3, axis=2)
    fed = scramble_patches(img)
    src = fed[::PATCH_SIZE, ::PATCH_SIZE, 0].astype(np.int64).reshape(-1)
    assert np.array_equal(np.sort(src), np.arange(grid_h * grid_w)), "not a permutation"
    return src


def align_to_true_geometry(tokens_fed, src):
    """Scatter fed-order tokens onto true patch positions."""
    out = np.empty_like(tokens_fed)
    out[src] = tokens_fed
    return out


def patch_means(img_hw3):
    """Per-patch channel mean, raster order -- the identity-check's stand-in features."""
    h, w, c = img_hw3.shape
    gh, gw = h // PATCH_SIZE, w // PATCH_SIZE
    return (img_hw3.reshape(gh, PATCH_SIZE, gw, PATCH_SIZE, c)
            .mean(axis=(1, 3))
            .reshape(gh * gw, c))


# -------------------------------------------------------------------------- models


def load_dino(weights_path, device, dtype):
    import torch

    from qwenvl.nautilus_model.dinov2 import DINOv2

    model = DINOv2("vitl")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def load_w1_mlp(checkpoint_dir, device, dtype):
    """``visual.nautilus_w1_mlp`` read straight out of the checkpoint shards."""
    import torch
    from safetensors import safe_open

    from qwenvl.nautilus_model.Nautilus_layers import MLP

    checkpoint_dir = Path(checkpoint_dir)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = "visual.nautilus_w1_mlp."

    by_shard = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise SystemExit(f"no {prefix}* tensors in {checkpoint_dir}")

    state = {}
    for shard, keys in by_shard.items():
        with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
            for key in keys:
                state[key[len(prefix):]] = f.get_tensor(key).float()

    in_dim = state["fc1.weight"].shape[1]
    hidden = state["fc1.weight"].shape[0]
    out_dim = state["fc2.weight"].shape[0]
    mlp = MLP(in_dim, [hidden], out_dim)
    mlp.load_state_dict(state, strict=True)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp.to(device=device, dtype=dtype), (in_dim, hidden, out_dim)


def dino_tokens(model, img_hw3, device, dtype):
    """(N, 1024) layer-23 patch tokens, in the fed image's raster order."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(img_hw3.transpose(2, 0, 1)))
    x = x.to(device=device, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        out = model.get_intermediate_layers(
            x, [DINO_LAYER], return_class_token=False
        )[0]
    return out.squeeze(0).float().cpu().numpy()


def w1_of(mlp, tokens, device, dtype):
    import torch

    x = torch.from_numpy(tokens).to(device=device, dtype=dtype)
    with torch.no_grad():
        return mlp(x).float().cpu().numpy()


# ------------------------------------------------------------------------- metrics


def cosine_per_patch(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def gain_relative_difference(w1_a, w1_b):
    """Per-token ||exp(w1_a) - exp(w1_b)|| / ||exp(w1_b)||.

    ``weight_2 = 1/exp(-w1)`` multiplies Qwen's 1280-d visual token elementwise
    (``ehance_embeds``), so this is how much the scramble changes what the LLM is
    handed -- not how much it changes DINOv2's own feature space.
    """
    ga, gb = np.exp(w1_a), np.exp(w1_b)
    return np.linalg.norm(ga - gb, axis=1) / (np.linalg.norm(gb, axis=1) + 1e-12)


def joint_pca_rgb(feat_b, feat_a):
    """Fit 3 PCs on both token sets together; return two (N, 3) arrays in 0..1."""
    both = np.concatenate([feat_b, feat_a], axis=0)
    mu = both.mean(axis=0)
    centred = both - mu
    cov = (centred.T @ centred) / len(centred)
    _, vecs = np.linalg.eigh(cov)
    comps = vecs[:, ::-1][:, :3].T  # (3, D), largest eigenvalue first

    proj = centred @ comps.T
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    scaled = np.clip((proj - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return scaled[: len(feat_b)], scaled[len(feat_b):]


# ------------------------------------------------------------------------- figures


def _grid(values, grid_h, grid_w):
    return np.asarray(values).reshape(grid_h, grid_w, -1).squeeze()


def render_frame(out_dir, stem, resized, feat_b, feat_a, cos, w1_b, w1_a,
                 grid_h, grid_w):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ref = np.clip(resized * 255.0, 0, 255).astype(np.uint8)

    # --- pca.png
    pca_b, pca_a = joint_pca_rgb(feat_b, feat_a)
    fig, axes = plt.subplots(3, 1, figsize=(11, 13.5), constrained_layout=True)
    axes[0].imshow(ref)
    axes[0].set_title(f"{stem}\nsmart_resize reference ({grid_h}x{grid_w} patches)")
    axes[1].imshow(_grid(pca_b, grid_h, grid_w), interpolation="nearest")
    axes[1].set_title("DINOv2 features, correct input (joint PCA -> RGB)")
    axes[2].imshow(_grid(pca_a, grid_h, grid_w), interpolation="nearest")
    axes[2].set_title("DINOv2 features, as-shipped scrambled input, "
                      "re-aligned to true geometry")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_dir / "pca.png", dpi=110)
    plt.close(fig)

    # --- cosine.png
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.6]})
    axes[0].imshow(ref)
    axes[0].set_title("frame")
    lo = min(0.0, float(cos.min()))
    im = axes[1].imshow(_grid(cos, grid_h, grid_w), cmap="RdYlBu",
                        vmin=lo, vmax=1.0, interpolation="nearest")
    axes[1].set_title("cos(as-shipped, correct) per patch")
    fig.colorbar(im, ax=axes[1], fraction=0.035)
    axes[2].hist(cos, bins=60, range=(lo, 1.0), color="#4477aa")
    axes[2].axvline(float(np.median(cos)), color="crimson", lw=1.2,
                    label=f"median {np.median(cos):.3f}")
    axes[2].set_title("distribution")
    axes[2].legend()
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_dir / "cosine.png", dpi=110)
    plt.close(fig)

    # --- gain.png  (RMS over the 1280 channels: the plain mean cancels itself out)
    rb = np.sqrt((w1_b ** 2).mean(axis=1))
    ra = np.sqrt((w1_a ** 2).mean(axis=1))
    reldiff = gain_relative_difference(w1_a, w1_b)
    vmin = float(min(rb.min(), ra.min()))
    vmax = float(max(rb.max(), ra.max()))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, data, title in (
        (axes[0], rb, "RMS w1, correct input"),
        (axes[1], ra, "RMS w1, as-shipped"),
    ):
        im = ax.imshow(_grid(data, grid_h, grid_w), cmap="viridis",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.035)
    im = axes[2].imshow(_grid(reldiff, grid_h, grid_w), cmap="magma",
                        interpolation="nearest")
    axes[2].set_title("relative change of the applied gain\n"
                      "||exp(w1_a) - exp(w1_b)|| / ||exp(w1_b)||")
    fig.colorbar(im, ax=axes[2], fraction=0.035)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{stem} -- w1 feeds weight_2 = exp(w1), the per-token gain on "
                 f"Qwen's visual tokens")
    fig.savefig(out_dir / "gain.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- gates


def identity_check(normed):
    """The alignment scatter must be exactly invertible on raw patch content."""
    gh, gw = normed.shape[0] // PATCH_SIZE, normed.shape[1] // PATCH_SIZE
    src = build_src_permutation(gh, gw)
    true_means = patch_means(normed)
    fed_means = patch_means(scramble_patches(normed))
    aligned = align_to_true_geometry(fed_means, src)
    max_dev = float(np.abs(aligned - true_means).max())
    ok = max_dev == 0.0
    print(f"[identity-check] grid {gh}x{gw}  max|aligned - true| = {max_dev:.3e}  "
          f"-> {'OK' if ok else 'FAILED'}")
    return ok


def noise_floor(model, normed, device, dtype):
    a = dino_tokens(model, normed, device, dtype)
    b = dino_tokens(model, normed, device, dtype)
    cos = cosine_per_patch(a, b)
    print(f"[noise-floor] identical input twice: bit-identical={np.array_equal(a, b)}  "
          f"cos min={cos.min():.6f} mean={cos.mean():.6f}")
    return cos


# ---------------------------------------------------------------------------- main


def resolve_stems(args):
    if args.images:
        return [Path(p) for p in args.images]
    subsample = json.loads(Path(args.subsample).read_text())
    split = subsample.get("split", "test")
    images_dir = Path(args.dataset) / split / "images"
    paths = []
    for stem in subsample["stems"]:
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                paths.append(candidate)
                break
        else:
            raise SystemExit(f"frame not found for stem {stem} under {images_dir}")
    return paths


def read_done(csv_path):
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {row["stem"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/workspace/runs/dino_scramble")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--subsample", default=None,
                    help="default: <dataset>/screening_subsample.json")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit frame paths instead of the pinned subsample")
    ap.add_argument("--dino-weights", default=DEFAULT_DINO_WEIGHTS)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                    help="bf16 matches how batch_inference.py loads the model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render", type=int, default=8,
                    help="write figures for the first N frames processed")
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--identity-check", action="store_true")
    ap.add_argument("--noise-floor", action="store_true")
    args = ap.parse_args()

    if args.subsample is None:
        args.subsample = str(Path(args.dataset) / "screening_subsample.json")

    import torch

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metrics.csv"

    paths = resolve_stems(args)
    if args.limit:
        paths = paths[: args.limit]

    # gates run on the first frame, before anything expensive
    _, first_normed = load_frame(paths[0], args.min_pixels, args.max_pixels)
    if args.identity_check and not identity_check(first_normed):
        raise SystemExit("alignment permutation is wrong -- every later number is void")

    print(f"loading DINOv2 ViT-L/14 from {args.dino_weights} ({args.dtype}, {args.device})")
    dino = load_dino(args.dino_weights, args.device, dtype)
    w1_mlp, dims = load_w1_mlp(args.checkpoint, args.device, dtype)
    print(f"loaded nautilus_w1_mlp: {dims[0]} -> {dims[1]} -> {dims[2]}")

    if args.noise_floor:
        noise_floor(dino, first_normed, args.device, dtype)

    done = read_done(csv_path)
    new_file = not csv_path.exists()
    rendered = 0
    rows = []

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()

        for i, path in enumerate(paths, 1):
            stem = path.stem
            fig_dir = out_root / "figures" / stem
            # a stem already in metrics.csv is still worth loading if it has no
            # figures yet and the render budget is unspent
            want_render = rendered < args.render and not (fig_dir / "gain.png").exists()
            if stem in done and not want_render:
                continue
            resized, normed = load_frame(path, args.min_pixels, args.max_pixels)
            gh = normed.shape[0] // PATCH_SIZE
            gw = normed.shape[1] // PATCH_SIZE
            src = build_src_permutation(gh, gw)

            feat_b = dino_tokens(dino, normed, args.device, dtype)
            feat_a_fed = dino_tokens(dino, scramble_patches(normed), args.device, dtype)
            feat_a = align_to_true_geometry(feat_a_fed, src)

            cos = cosine_per_patch(feat_a, feat_b)
            w1_b = w1_of(w1_mlp, feat_b, args.device, dtype)
            w1_a = w1_of(w1_mlp, feat_a, args.device, dtype)
            reldiff = gain_relative_difference(w1_a, w1_b)

            row = {
                "stem": stem,
                "grid_h": gh,
                "grid_w": gw,
                "n_patches": gh * gw,
                "cos_mean": round(float(cos.mean()), 6),
                "cos_median": round(float(np.median(cos)), 6),
                "cos_p05": round(float(np.percentile(cos, 5)), 6),
                "cos_frac_below_0.5": round(float((cos < 0.5).mean()), 6),
                "w1_rms_b": round(float(np.sqrt((w1_b ** 2).mean())), 6),
                "w1_rms_a": round(float(np.sqrt((w1_a ** 2).mean())), 6),
                "w1_abs_delta_mean": round(float(np.abs(w1_a - w1_b).mean()), 6),
                "gain_p01_b": round(float(np.exp(np.percentile(w1_b, 1))), 6),
                "gain_p99_b": round(float(np.exp(np.percentile(w1_b, 99))), 6),
                "gain_p01_a": round(float(np.exp(np.percentile(w1_a, 1))), 6),
                "gain_p99_a": round(float(np.exp(np.percentile(w1_a, 99))), 6),
                "gain_reldiff_mean": round(float(reldiff.mean()), 6),
                "gain_reldiff_median": round(float(np.median(reldiff)), 6),
            }
            if stem not in done:
                writer.writerow(row)
                fh.flush()
                rows.append(row)

            if want_render:
                render_frame(fig_dir, stem, resized,
                             feat_b, feat_a, cos, w1_b, w1_a, gh, gw)
                rendered += 1

            print(f"[{i}/{len(paths)}] {stem}  grid {gh}x{gw}  "
                  f"cos mean {row['cos_mean']:.4f} median {row['cos_median']:.4f}  "
                  f"|dw1| {row['w1_abs_delta_mean']:.4f}")

    if rows:
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in CSV_FIELDS if k not in ("stem", "grid_h", "grid_w", "n_patches")}
        summary = {
            "dataset": args.dataset,
            "subsample": args.subsample,
            "dino_weights": args.dino_weights,
            "checkpoint": args.checkpoint,
            "dtype": args.dtype,
            "device": args.device,
            "dino_layer": DINO_LAYER,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "normalisation": "qwen CLIP (both conditions)",
            "n_frames_this_run": len(rows),
            "means_this_run": {k: round(v, 6) for k, v in agg.items()},
        }
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["means_this_run"], indent=2))
    print(f"-> {csv_path}  ({len(rows)} new rows, {rendered} frames rendered)")


if __name__ == "__main__":
    main()
