"""VFE anatomy: ablate NAUTILUS' Visual Feature Enhancement end to end.

The VFE as implemented (``Qwen2_5_VL_Nautilus_ForConditionalGeneration.py``,
``ehance_embeds`` + ``forward``)::

    v    = Qwen ViT output                      (seq x 1280, window order)
    w1   = nautilus_w1_mlp(DINOv2(restore_image_from_patches(pixels)))
    s    = nautilus_dark_mlp(v[darkest patch] - dark_attn(global_queries + mean v, v))
    enh  = exp(w1) * (v - s)
    LLM <- interleave(merger(v), merger(enh))   (token doubling)

Every condition below is applied to one loaded model through forward hooks and
instance-attribute wrappers only -- the model file is never touched:

  full                       nothing                               exp(w1)*(v-s)
  no_gain                    w1_mlp output -> zeros                v-s
  no_backscatter             dark_mlp output -> zeros              exp(w1)*v
  vfe_identity               both                                  v (each token twice)
  dino_unscrambled           correct block->raster restore, and    exp(w1')*(v-s)
                             DINOv2's raster tokens permuted back
                             into Qwen block order
  dino_unscrambled_imagenet  as above + CLIP->ImageNet renorm      same, ImageNet stats

**Why the token permutation.** ``ehance_embeds`` applies ``window_index`` to DINOv2's
tokens on the assumption that DINOv2 token *n* is Qwen sequence token *n* -- which the
shipped (scrambled) restore happens to satisfy, since it lays sequence patch *n* into
raster cell *n*. Feeding DINOv2 the correct raster image breaks that: token *n* would
then be raster patch *n*, and every gain would land on the wrong Qwen token with no
error anywhere. ``get_intermediate_layers`` is therefore wrapped to reorder its output
with ``idx[n] = (2*br+ir)*gw + (2*bc+ic)``. ``--self-test`` proves both halves against
Qwen's real image processor before any GPU time is spent. The dark-patch brightness
reads ``pixel_single_values`` directly, so none of the wrappers affect it.

Modes
-----
``--self-test``   CPU, no weights. Id-encoded image -> ``Qwen2VLImageProcessor`` ->
                  shipped restore must equal ``scramble_patches``, wrapped restore must
                  equal the raster id image, and ``idx`` must map raster tokens back
                  onto sequence ids exactly.
``--measure``     Visual tower only (``model.visual``), no generation, over the
                  250-frame screening subsample, all six conditions per frame.
                  Writes ``components.csv`` (backscatter size/direction, the dark patch,
                  enhancement split into gain and backscatter terms, what the LLM gets,
                  how far each ablation moves the LLM input) and enforces two bit-exact
                  gates: ``vfe_identity`` -> merged_enh == merged_v, ``no_gain`` ->
                  enh == v - s. ``--render N`` adds ``vfe.png`` per frame.
default           Generation over the full test split through
                  ``prompt_experiments.run_messages`` with ``P0_baseline`` -- the exact
                  P0 query path (true grid from ``image_grid_thw``, token doubling).
                  Output ``<runs>/<condition><tag>/{results,results_raw,metadata.json}``.
``--report``      CPU. ``prompt_report.py`` table + ``--diff full <cond>`` per condition,
                  then a paired video-cluster bootstrap of the deltas vs ``full``.
                  Writes ``<runs>/ablation_report.{json,md}``.

Usage:
    # 0. the index algebra, against Qwen's own processor (CPU)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --self-test"

    # 1. component measurement + bit-exact gates, 8 figures (one GPU, ~10 min)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --measure --render 8 --device 0"

    # 2. determinism gates, 20 subsample images each
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --conditions full --subsample --limit 20 \\
         --runs /workspace/runs/vfe_ablation_gates --tag _cuda0 --device 0 && \\
       python3 vfe_ablation.py --conditions full --subsample --limit 20 --hook-noop \\
         --runs /workspace/runs/vfe_ablation_gates --tag _noop_cuda0 --device 0"
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --conditions full --subsample --limit 20 \\
         --runs /workspace/runs/vfe_ablation_gates --tag _cuda1 --device 1"
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 prompt_report.py --diff /workspace/runs/vfe_ablation_gates/full_cuda0 \\
                                       /workspace/runs/vfe_ablation_gates/full_noop_cuda0"

    # 3. the sweep, one shard per GPU (resumable)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --conditions full,dino_unscrambled --device 0 --skip-existing"

    # 4. the report (CPU)
    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts/thuenen_pipeline && \\
       python3 vfe_ablation.py --report"
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

import numpy as np  # noqa: E402

from nautilus_zeroshot import video_id_of  # noqa: E402
from prompt_experiments import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_ROOT,
    fingerprint,
    load_subsample,
    query_path_for,
    read_prompt_classes,
)
from visualize_dino_input import (  # noqa: E402
    MERGE_SIZE,
    PATCH_SIZE,
    QWEN_MEAN,
    QWEN_STD,
    scramble_patches,
)

DEFAULT_RUNS = "/workspace/runs/vfe_ablation"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)

CONDITIONS = ["full", "no_gain", "no_backscatter", "vfe_identity",
              "dino_unscrambled", "dino_unscrambled_imagenet"]
ABLATIONS = CONDITIONS[1:]
ZERO_GAIN = {"no_gain", "vfe_identity"}
ZERO_BACKSCATTER = {"no_backscatter", "vfe_identity"}
UNSCRAMBLE = {"dino_unscrambled": False, "dino_unscrambled_imagenet": True}  # -> imagenet
VARIANT = "P0_baseline"


# ── index algebra (numpy, torch-free) ────────────────────────────────────────
def raster_to_block_index(grid_h, grid_w):
    """``idx[n]`` = raster patch index of Qwen sequence token ``n``.

    Qwen's sequence order is ``(block_row, block_col, in_row, in_col)``. Gathering a
    raster-order token array with ``idx`` puts it into sequence order.
    """
    m = MERGE_SIZE
    br, bc, ir, ic = np.meshgrid(np.arange(grid_h // m), np.arange(grid_w // m),
                                 np.arange(m), np.arange(m), indexing="ij")
    return ((m * br + ir) * grid_w + (m * bc + ic)).reshape(-1)


def restore_correct(patches, h_patches, w_patches):
    """Block-order ``(N, C, p, p)`` patches -> the true ``(C, gh*p, gw*p)`` image."""
    n, c, p, _ = patches.shape
    gh, gw = int(h_patches), int(w_patches)
    assert n == gh * gw, "patch count does not match the grid"
    m = MERGE_SIZE
    x = patches.reshape(gh // m, gw // m, m, m, c, p, p)
    return x.permute(4, 0, 2, 5, 1, 3, 6).reshape(c, gh * p, gw * p)


def clip_to_imagenet(image):
    """CLIP-normalised ``(C, H, W)`` -> ImageNet-normalised, computed in fp32."""
    import torch

    dtype = image.dtype
    x = image.float()
    as_t = lambda a: torch.tensor(a, dtype=torch.float32, device=x.device).view(3, 1, 1)  # noqa: E731
    x = (x * as_t(QWEN_STD) + as_t(QWEN_MEAN) - as_t(IMAGENET_MEAN)) / as_t(IMAGENET_STD)
    return x.to(dtype)


# ── condition machinery ──────────────────────────────────────────────────────
class ConditionControl(object):
    """Applies one condition to ``model.visual`` and removes it again.

    Hooks are registered before any capture hook, so a capture hook on the same
    module sees the *replaced* output -- what the model actually consumes.
    """

    def __init__(self, visual):
        self.visual = visual
        self.handles = []
        self.wrapped = []  # (object, attribute name)

    def _zero_hook(self, module):
        import torch
        self.handles.append(module.register_forward_hook(
            lambda _m, _i, out: torch.zeros_like(out)))

    def _noop_hook(self, module):
        self.handles.append(module.register_forward_hook(lambda _m, _i, _o: None))

    def _wrap(self, obj, name, fn):
        setattr(obj, name, fn)
        self.wrapped.append((obj, name))

    def apply(self, condition, hook_noop=False):
        import torch

        self.clear()
        visual = self.visual
        if condition not in CONDITIONS:
            raise SystemExit("unknown condition {}".format(condition))
        if hook_noop:
            if condition != "full":
                raise SystemExit("--hook-noop only makes sense with the full condition")
            # Pass-through versions of every mechanism the ablations use, so the gate
            # tests the machinery itself rather than just an unhooked model.
            self._noop_hook(visual.nautilus_w1_mlp)
            self._noop_hook(visual.nautilus_dark_mlp)
            shipped_restore = visual.restore_image_from_patches
            shipped_gil = visual.nautilus_encoder.get_intermediate_layers
            self._wrap(visual, "restore_image_from_patches",
                       lambda patches, h, w: shipped_restore(patches, h, w))
            self._wrap(visual.nautilus_encoder, "get_intermediate_layers",
                       lambda *a, **k: shipped_gil(*a, **k))
            return

        if condition in ZERO_GAIN:
            self._zero_hook(visual.nautilus_w1_mlp)
        if condition in ZERO_BACKSCATTER:
            self._zero_hook(visual.nautilus_dark_mlp)
        if condition in UNSCRAMBLE:
            imagenet = UNSCRAMBLE[condition]
            shipped_gil = visual.nautilus_encoder.get_intermediate_layers

            def restore(patches, h_patches, w_patches):
                image = restore_correct(patches, h_patches, w_patches)
                return clip_to_imagenet(image) if imagenet else image

            def gil(x, n=1, reshape=False, return_class_token=False, norm=True):
                if reshape or return_class_token:
                    raise NotImplementedError("wrapper only covers ehance_embeds' call")
                outputs = shipped_gil(x, n, reshape=False, return_class_token=False,
                                      norm=norm)
                gh, gw = x.shape[-2] // PATCH_SIZE, x.shape[-1] // PATCH_SIZE
                idx = torch.as_tensor(raster_to_block_index(gh, gw), device=x.device)
                return tuple(out[:, idx] for out in outputs)

            self._wrap(visual, "restore_image_from_patches", restore)
            self._wrap(visual.nautilus_encoder, "get_intermediate_layers", gil)

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for obj, name in self.wrapped:
            delattr(obj, name)  # the class method shows through again
        self.wrapped = []


def load_model(checkpoint, device, query_max_pixels):
    """Same load as ``prompt_experiments.run_variants``."""
    sys.path.insert(0, os.path.dirname(SCRIPTS))
    import torch
    from transformers import AutoProcessor

    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_VL_Nautilus_ForConditionalGeneration,
    )

    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        checkpoint, cache_dir=None, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, device_map="cuda:" + str(device))
    model.eval()
    processor = AutoProcessor.from_pretrained(
        checkpoint, min_pixels=1 * 28 * 28, max_pixels=query_max_pixels * 28 * 28)
    return model, processor


def make_context(args, prompt_names):
    from prompt_variants import Context
    return Context(prompt_names, args.query_max_pixels * 28 * 28,
                   args.exemplar_max_pixels * 28 * 28)


def resolve_stems(args):
    """Full test split by default; the pinned screening subsample on request."""
    if args.subsample or args.measure:
        stems = list(load_subsample(args.root)["stems"])
    else:
        images_dir = os.path.join(args.root, args.split, "images")
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                       if f.lower().endswith(".jpg"))
    return stems[:args.limit] if args.limit else stems


# ── --self-test ──────────────────────────────────────────────────────────────
def self_test(args):
    sys.path.insert(0, os.path.dirname(SCRIPTS))
    from types import SimpleNamespace

    import torch
    from transformers import Qwen2VLImageProcessor

    from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import (
        Qwen2_5_Nautilus_VisionTransformerPretrainedModel as Visual,
    )

    processor = Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=4096 * 28 * 28)
    ok = True
    for gh, gw in ((6, 8), (54, 96), (54, 98)):
        p = PATCH_SIZE
        ids = np.arange(gh * gw, dtype=np.float32).reshape(gh, gw)
        # Each patch a different value per channel, so a channel mix-up fails too.
        raster = np.stack([ids, ids + 0.25, ids + 0.5], axis=-1)
        raster = np.repeat(np.repeat(raster, p, axis=0), p, axis=1)  # (H, W, 3)

        out = processor(images=[raster], do_resize=False, do_rescale=False,
                        do_normalize=False, return_tensors="pt",
                        input_data_format="channels_last")
        grid = out["image_grid_thw"][0].tolist()
        assert grid == [1, gh, gw], "processor grid {} != {}".format(grid, [1, gh, gw])
        pixel_values = out["pixel_values"]
        # exactly what ehance_embeds does before restoring
        single = pixel_values.reshape(-1, 3, 2, p, p)[:, :, 0, :, :]

        shipped = Visual.restore_image_from_patches(None, single, gh, gw)
        wrapped = restore_correct(single, gh, gw)
        raster_t = torch.from_numpy(raster.transpose(2, 0, 1).copy())
        scrambled_t = torch.from_numpy(scramble_patches(raster.astype(np.float64))
                                       .transpose(2, 0, 1).astype(np.float32).copy())

        checks = {
            "wrapped restore == raster id image": torch.equal(wrapped, raster_t),
            "shipped restore == scramble_patches(raster)": torch.equal(shipped, scrambled_t),
        }
        # DINOv2 stand-in: raster-order "tokens" = per-patch mean of the wrapped image,
        # i.e. each token carries its raster id. After gathering with idx, token n must
        # carry the id of Qwen sequence patch n.
        raster_tokens = wrapped[0].reshape(gh, p, gw, p).mean(dim=(1, 3)).reshape(-1)
        seq_ids = single[:, 0].mean(dim=(1, 2))
        idx = torch.as_tensor(raster_to_block_index(gh, gw))
        checks["idx maps raster tokens onto sequence ids"] = torch.equal(
            raster_tokens[idx], seq_ids)
        # and the shipped path is aligned by construction (token n = sequence patch n)
        shipped_tokens = shipped[0].reshape(gh, p, gw, p).mean(dim=(1, 3)).reshape(-1)
        checks["shipped tokens already in sequence order"] = torch.equal(
            shipped_tokens, seq_ids)
        checks["idx is a permutation"] = bool(
            np.array_equal(np.sort(idx.numpy()), np.arange(gh * gw)))

        # --measure's figure mapping: window-ordered values (as the ViT reorders them)
        # must land back on their raster cells, via the model's own get_window_index
        fake = SimpleNamespace(window_size=112, spatial_merge_size=MERGE_SIZE,
                               patch_size=p, spatial_merge_unit=MERGE_SIZE ** 2)
        window_index, _ = Visual.get_window_index(fake, torch.tensor([[1, gh, gw]]))
        window_index = torch.as_tensor(window_index)
        windowed = seq_ids.reshape(-1, MERGE_SIZE ** 2)[window_index].reshape(-1)
        back = window_to_raster(windowed.unsqueeze(1), window_index, gh, gw).squeeze(1)
        checks["window_to_raster undoes window order"] = torch.equal(back, raster_tokens)

        # dark_patch must find one planted dark patch at a known raster cell
        r0, c0 = gh // 3, (2 * gw) // 3 + 1
        planted = np.ones((gh * p, gw * p, 3), dtype=np.float32)
        planted[r0 * p:(r0 + 1) * p, c0 * p:(c0 + 1) * p] = 0.0
        planted_pv = processor(images=[planted], do_resize=False, do_rescale=False,
                               do_normalize=False, return_tensors="pt",
                               input_data_format="channels_last")["pixel_values"]
        found = dark_patch(planted_pv, window_index, gh, gw)
        checks["dark_patch finds the planted patch"] = found[:2] == (r0, c0)

        # ImageNet renorm round trip on a constant grey
        grey = torch.full((3, 2, 2), 0.5)
        clip = (grey - torch.tensor(QWEN_MEAN, dtype=torch.float32).view(3, 1, 1)) / \
            torch.tensor(QWEN_STD, dtype=torch.float32).view(3, 1, 1)
        expect = (grey - torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)) / \
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        checks["clip_to_imagenet"] = bool(torch.allclose(clip_to_imagenet(clip), expect,
                                                         atol=1e-6))

        for name, passed in checks.items():
            print("[self-test] grid {}x{}  {:45s} {}".format(
                gh, gw, name, "OK" if passed else "FAILED"))
            ok &= bool(passed)

    # ConditionControl on a dummy object: wrappers must come off again
    class Dummy(object):
        def restore_image_from_patches(self, *a):
            return "shipped"

    dummy = Dummy()
    dummy.nautilus_encoder = Dummy()
    dummy.nautilus_encoder.get_intermediate_layers = None  # instance attr, not a method
    del dummy.nautilus_encoder.get_intermediate_layers
    dummy.nautilus_encoder.__class__.get_intermediate_layers = lambda self, *a, **k: "gil"
    control = ConditionControl(dummy)
    control.apply("dino_unscrambled")
    swapped = dummy.restore_image_from_patches is not Dummy.restore_image_from_patches
    control.clear()
    restored = dummy.restore_image_from_patches() == "shipped" and \
        "restore_image_from_patches" not in vars(dummy)
    print("[self-test] wrappers installed and removed cleanly            {}".format(
        "OK" if swapped and restored else "FAILED"))
    ok &= swapped and restored

    print("[self-test] {}".format("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# ── --measure ────────────────────────────────────────────────────────────────
MEASURE_FIELDS = [
    "stem", "grid_h", "grid_w",
    # backscatter
    "s_norm_over_median_v", "cos_s_mean_v",
    "dark_row", "dark_col", "dark_brightness_clip",
    # enhancement at ViT width (full)
    "enh_rel_median", "gain_term_rel_median", "backscatter_term_rel_median",
    "w1_rms", "gain_p01", "gain_p99",
    # what the LLM gets (full)
    "merged_cos_v_enh_median", "merged_cos_v_enh_p05", "merged_relnorm_diff_median",
    # condition vs full
    *["cos_vs_full_{}_{}".format(c, s) for c in ABLATIONS for s in ("median", "p05")],
    "w1_pearson_centred_vs_full_dino_unscrambled",
    "w1_pearson_centred_vs_full_dino_unscrambled_imagenet",
    "gain_reldiff_vs_full_dino_unscrambled",
    "gain_reldiff_vs_full_dino_unscrambled_imagenet",
    # gates
    "gate_identity_bitexact", "gate_no_gain_bitexact", "gate_v_invariant",
]


class Capture(object):
    """Forward hooks recording the VFE's intermediate tensors for one visual pass."""

    def __init__(self, visual):
        self.visual = visual
        self.handles = []
        self.reset()

    def reset(self):
        self.w1 = None
        self.s = None
        self.merger_in = []
        self.merger_out = []

    def attach(self):
        v = self.visual

        def w1_hook(_m, _i, out):
            self.w1 = out.float()

        # Native dtype (bf16): the bit-exact gates have to compare what the model
        # computed, and v.float() - s.float() rounds differently from (v - s).float().
        def s_hook(_m, _i, out):
            self.s = out

        def merger_hook(_m, inputs, out):
            self.merger_in.append(inputs[0])
            self.merger_out.append(out)

        self.handles = [v.nautilus_w1_mlp.register_forward_hook(w1_hook),
                        v.nautilus_dark_mlp.register_forward_hook(s_hook),
                        v.merger.register_forward_hook(merger_hook)]

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def torch_quantile(x, q):
    import torch
    return float(torch.quantile(x.float().flatten()[:16_000_000], q))


def centred_pearson_t(x, y):
    xc = x - x.mean(dim=0)
    yc = y - y.mean(dim=0)
    return float((xc * yc).sum() / ((xc ** 2).sum().sqrt() * (yc ** 2).sum().sqrt() + 1e-30))


def dark_patch(pixel_values, window_index, grid_h, grid_w):
    """The token ``ehance_embeds`` picks as darkest, located on the raster grid.

    Reproduces the model's argmin in *window* order, so ties (black borders) resolve
    exactly as they do inside the model.
    """
    import torch

    m2 = MERGE_SIZE * MERGE_SIZE
    brightness = pixel_values.float().reshape(-1, 3, 2, PATCH_SIZE, PATCH_SIZE)[
        :, :, 0].mean(dim=(1, 2, 3))
    windowed = brightness.reshape(-1, m2)[window_index].reshape(-1)
    t_win = int(torch.argmin(windowed))
    seq_token = int(window_index[t_win // m2]) * m2 + t_win % m2
    raster = int(raster_to_block_index(grid_h, grid_w)[seq_token])
    row, col = divmod(raster, grid_w)
    return row, col, float(brightness[seq_token])


def window_to_raster(tokens, window_index, grid_h, grid_w):
    """Unmerged window-order ``(seq, D)`` -> raster-order ``(seq, D)``."""
    import torch

    m2 = MERGE_SIZE * MERGE_SIZE
    reverse = torch.argsort(window_index)
    seq = tokens.reshape(-1, m2, tokens.shape[-1])[reverse].reshape(-1, tokens.shape[-1])
    out = torch.empty_like(seq)
    idx = torch.as_tensor(raster_to_block_index(grid_h, grid_w), device=tokens.device)
    out[idx] = seq
    return out


def render_vfe(out_path, stem, pixel_values, grid_h, grid_w, dark, rel_raster,
               merged_cos_grid):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    single = pixel_values.float().reshape(-1, 3, 2, PATCH_SIZE, PATCH_SIZE)[:, :, 0]
    image = restore_correct(single, grid_h, grid_w).cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * QWEN_STD + QWEN_MEAN, 0.0, 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
    axes[0].imshow(image)
    row, col, bright = dark
    axes[0].add_patch(mpatches.Rectangle(
        (col * PATCH_SIZE - 0.5, row * PATCH_SIZE - 0.5), PATCH_SIZE, PATCH_SIZE,
        fill=False, edgecolor="red", lw=2))
    axes[0].add_patch(mpatches.Circle(
        ((col + 0.5) * PATCH_SIZE, (row + 0.5) * PATCH_SIZE), 5 * PATCH_SIZE,
        fill=False, edgecolor="red", lw=1.2, ls="--"))
    axes[0].set_title("{}\nmodel input; red = the patch the backscatter estimate is "
                      "read from (row {}, col {}, CLIP brightness {:.2f})".format(
                          stem, row, col, bright))
    im = axes[1].imshow(rel_raster.reshape(grid_h, grid_w), cmap="magma",
                        interpolation="nearest")
    axes[1].set_title("||enh - v|| / ||v|| per ViT patch (full VFE)")
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    im = axes[2].imshow(merged_cos_grid, cmap="viridis", interpolation="nearest")
    axes[2].set_title("cos(merger(v), merger(enh)) per LLM token -- how different the "
                      "doubled token actually is")
    fig.colorbar(im, ax=axes[2], fraction=0.03)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def measure(args):
    from pathlib import Path

    import torch
    from tqdm import tqdm

    from prompt_experiments import build_inputs
    from prompt_variants import VARIANTS

    out_dir = Path(args.runs) / "measure"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "components.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != MEASURE_FIELDS:
                raise SystemExit("{} has a different header -- use a fresh --runs".format(
                    csv_path))
            done = {row["stem"] for row in reader}

    stems = resolve_stems(args)
    prompt_names = read_prompt_classes(args.root)
    images_dir = os.path.join(args.root, args.split, "images")
    model, processor = load_model(args.checkpoint, args.device, args.query_max_pixels)
    visual = model.visual
    ctx = make_context(args, prompt_names)
    control = ConditionControl(visual)
    capture = Capture(visual)
    spec = VARIANTS[VARIANT]

    failures = Counter()
    rendered = 0
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=MEASURE_FIELDS)
        if new_file:
            writer.writeheader()
        for stem in tqdm(stems, desc="measure"):
            fig_path = out_dir / "figures" / stem / "vfe.png"
            want_render = rendered < args.render and not fig_path.exists()
            if stem in done and not want_render:
                continue
            query = query_path_for(images_dir, stem)
            _, inputs = build_inputs(processor, spec["build"](ctx, query))
            pixel_values = inputs["pixel_values"].to(model.device).type(visual.dtype)
            grid_thw = inputs["image_grid_thw"].to(model.device)
            _, grid_h, grid_w = (int(v) for v in grid_thw[0])
            window_index, _ = visual.get_window_index(grid_thw)
            window_index = torch.as_tensor(window_index, device=model.device)

            per = {}
            for condition in CONDITIONS:
                control.apply(condition)
                capture.reset()
                capture.attach()
                visual(pixel_values, grid_thw=grid_thw)
                capture.detach()
                control.clear()
                assert len(capture.merger_in) == 2, "merger was called {} times".format(
                    len(capture.merger_in))
                per[condition] = {
                    "w1": capture.w1, "s": capture.s,
                    "v": capture.merger_in[0], "enh": capture.merger_in[1],
                    "mv": capture.merger_out[0], "menh": capture.merger_out[1],
                }

            full = per["full"]
            v, enh, s, w1 = (full[k].float() for k in ("v", "enh", "s", "w1"))
            mv, menh = full["mv"].float(), full["menh"].float()
            v_norm = v.norm(dim=1)
            s_vec = s.reshape(-1)
            gain = torch.exp(w1)
            gain_term = ((gain - 1.0) * (v - s)).norm(dim=1) / v_norm
            mean_v = v.mean(dim=0)
            mcos = torch.nn.functional.cosine_similarity(mv, menh, dim=1)
            mrel = (menh.norm(dim=1) - mv.norm(dim=1)) / mv.norm(dim=1)
            dark = dark_patch(pixel_values, window_index, grid_h, grid_w)

            row = {
                "stem": stem, "grid_h": grid_h, "grid_w": grid_w,
                "s_norm_over_median_v": s_vec.norm() / v_norm.median(),
                "cos_s_mean_v": torch.nn.functional.cosine_similarity(s_vec, mean_v, dim=0),
                "dark_row": dark[0], "dark_col": dark[1], "dark_brightness_clip": dark[2],
                "enh_rel_median": ((enh - v).norm(dim=1) / v_norm).median(),
                "gain_term_rel_median": gain_term.median(),
                "backscatter_term_rel_median": (s_vec.norm() / v_norm).median(),
                "w1_rms": (w1 ** 2).mean().sqrt(),
                "gain_p01": np.exp(torch_quantile(w1, 0.01)),
                "gain_p99": np.exp(torch_quantile(w1, 0.99)),
                "merged_cos_v_enh_median": mcos.median(),
                "merged_cos_v_enh_p05": torch.quantile(mcos, 0.05),
                "merged_relnorm_diff_median": mrel.median(),
            }
            for condition in ABLATIONS:
                cos = torch.nn.functional.cosine_similarity(
                    per[condition]["menh"].float(), menh, dim=1)
                row["cos_vs_full_{}_median".format(condition)] = cos.median()
                row["cos_vs_full_{}_p05".format(condition)] = torch.quantile(cos, 0.05)
            for condition in UNSCRAMBLE:
                w1c = per[condition]["w1"]
                row["w1_pearson_centred_vs_full_" + condition] = centred_pearson_t(w1c, w1)
                row["gain_reldiff_vs_full_" + condition] = (
                    (torch.exp(w1c) - gain).norm(dim=1) / gain.norm(dim=1)).mean()

            ident = per["vfe_identity"]
            nogain = per["no_gain"]
            gates = {
                "gate_identity_bitexact": bool(torch.equal(ident["menh"], ident["mv"])),
                "gate_no_gain_bitexact": bool(torch.equal(nogain["enh"],
                                                          nogain["v"] - nogain["s"])),
                "gate_v_invariant": all(torch.equal(per[c]["v"], full["v"])
                                        for c in CONDITIONS),
            }
            row.update(gates)
            for name, passed in gates.items():
                if not passed:
                    failures[name] += 1
                    print("[gate FAILED] {} {}".format(stem, name))

            row = {k: (round(float(val), 6) if isinstance(val, (float, torch.Tensor,
                                                                 np.floating)) else val)
                   for k, val in row.items()}
            if stem not in done:
                writer.writerow(row)
                handle.flush()

            if want_render:
                rel = (enh - v).norm(dim=1) / v_norm
                rel_raster = window_to_raster(rel.unsqueeze(1), window_index,
                                              grid_h, grid_w).squeeze(1).cpu().numpy()
                reverse = torch.argsort(window_index)
                mcos_grid = mcos[reverse].reshape(grid_h // MERGE_SIZE,
                                                  grid_w // MERGE_SIZE).cpu().numpy()
                render_vfe(fig_path, stem, pixel_values, grid_h, grid_w, dark,
                           rel_raster, mcos_grid)
                rendered += 1

    summarise_measure(csv_path, out_dir / "summary.json", failures)
    if failures:
        print("[measure] GATES FAILED: {}".format(dict(failures)))
        return 1
    print("[measure] all bit-exact gates passed")
    return 0


def summarise_measure(csv_path, out_path, failures):
    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    summary = {"frames": len(rows), "gate_failures_this_run": dict(failures),
               "gate_pass_counts": {}, "median_over_frames": {}, "mean_over_frames": {}}
    for key in MEASURE_FIELDS:
        if key.startswith("gate_"):
            summary["gate_pass_counts"][key] = sum(r[key] == "True" for r in rows)
        elif key not in ("stem", "grid_h", "grid_w", "dark_row", "dark_col"):
            values = [float(r[key]) for r in rows if r[key] != ""]
            summary["median_over_frames"][key] = round(float(np.median(values)), 6)
            summary["mean_over_frames"][key] = round(float(np.mean(values)), 6)
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


# ── default: inference ───────────────────────────────────────────────────────
def run_condition(args, model, processor, condition, stems, prompt_names, images_dir):
    """Mirror of ``prompt_experiments.run_variant`` for one VFE condition."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    from nautilus_zeroshot import snap_labels
    from prompt_experiments import run_messages
    from prompt_variants import VARIANTS

    spec = VARIANTS[VARIANT]
    name = condition + args.tag
    out_dir = os.path.join(args.runs, name)
    results_dir = os.path.join(out_dir, "results")
    raw_dir = os.path.join(out_dir, "results_raw")
    for directory in (results_dir, raw_dir):
        os.makedirs(directory, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.json")

    gpu_name = torch.cuda.get_device_name(model.device)
    previous = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    if previous.get("gpu_name") and previous["gpu_name"] != gpu_name:
        raise SystemExit(
            "{} was started on {} and would resume on {}. Greedy outputs differ across "
            "GPU generations, so a mixed run is not one condition.".format(
                name, previous["gpu_name"], gpu_name))
    if previous.get("hook_noop", args.hook_noop) != args.hook_noop:
        raise SystemExit("{}: --hook-noop differs from the run being resumed".format(name))

    metadata = {
        "prompt": previous.get("prompt"),
        "checkpoint": args.checkpoint,
        "image_dims": previous.get("image_dims", {}),
        "variant": VARIANT,
        "note": spec.get("note"),
        "subsample_sha1": fingerprint(stems),
        "subsample_n": len(stems),
        "exemplars": [],
        "exemplar_max_pixels": args.exemplar_max_pixels,
        "query_max_pixels": args.query_max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "snapped": True,
        "prompt_tokens": previous.get("prompt_tokens", {}),
        "condition": condition,
        "hook_noop": args.hook_noop,
        "split": "subsample" if args.subsample else args.split,
        "gpu_name": gpu_name,
        "device": "cuda:{}".format(args.device),
    }
    ctx = make_context(args, prompt_names)
    control = ConditionControl(model.visual)
    control.apply(condition, hook_noop=args.hook_noop)
    stats = Counter()
    try:
        for stem in tqdm(stems, desc=name):
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
                    model, processor, spec, ctx, [], query_path, args.max_new_tokens)
            except Exception as error:  # noqa: BLE001
                print("[failed] {}: {}".format(stem, error))
                stats["failed"] += 1
                continue
            if metadata["prompt"] is None:
                metadata["prompt"] = text

            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write(response)
            snapped, unmatched = snap_labels(response, prompt_names)
            stats["unmatched_labels"] += unmatched
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
        control.clear()
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print("{}: ok={} skipped={} failed={} unmatched_labels={} on {}".format(
        name, stats["ok"], stats["skipped"], stats["failed"],
        stats["unmatched_labels"], gpu_name))
    return stats


def run_inference(args):
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CONDITIONS]
    if unknown:
        raise SystemExit("unknown condition(s): {}\nknown: {}".format(
            ", ".join(unknown), ", ".join(CONDITIONS)))
    stems = resolve_stems(args)
    prompt_names = read_prompt_classes(args.root)
    images_dir = os.path.join(args.root, args.split, "images")
    print("{} conditions x {} images on cuda:{}".format(len(requested), len(stems),
                                                       args.device))
    model, processor = load_model(args.checkpoint, args.device, args.query_max_pixels)
    for condition in requested:
        run_condition(args, model, processor, condition, stems, prompt_names, images_dir)
    return 0


# ── --report ─────────────────────────────────────────────────────────────────
def per_image_matches(run_dir, args, class_names):
    """``{stem: (n_gt, n_pred, matched_centre, matched_iou50)}`` via the house scorers."""
    from pathlib import Path

    from evaluate_detections import greedy_match, load_dataset
    from localization_report import boxes_of, centre_in_box_match

    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as handle:
        image_dims = json.load(handle).get("image_dims", {})
    devnull = open(os.devnull, "w")
    stderr, sys.stderr = sys.stderr, devnull
    try:
        dataset, _ = load_dataset(
            Path(args.root) / args.split / "labels_prompt", Path(run_dir) / "results",
            Path(args.root) / args.split / "images", image_dims, class_names, False)
    finally:
        sys.stderr = stderr
        devnull.close()

    out = {}
    for stem, record in dataset.items():
        pred = boxes_of(record["pred"])
        gt = [d["bbox_2d"] for d in record["gt"]
              if isinstance(d.get("bbox_2d"), list) and len(d["bbox_2d"]) == 4]
        pairs, _, _ = greedy_match(pred, [None] * len(pred), gt, [None] * len(gt),
                                   0.5, class_aware=False)
        out[stem] = (len(gt), len(pred), len(centre_in_box_match(pred, gt)), len(pairs))
    return out


def paired_bootstrap(full, cond, n_resamples, seed):
    """Video-cluster bootstrap of cond - full on the stems both runs share."""
    stems = sorted(set(full) & set(cond))
    videos = sorted({video_id_of(s) for s in stems})
    col = {v: i for i, v in enumerate(videos)}
    # per-video sums: images, gt, then (pred, centre, iou50) for full and cond
    sums = np.zeros((len(videos), 8))
    for stem in stems:
        i = col[video_id_of(stem)]
        g, pf, cf, mf = full[stem]
        _, pc, cc, mc = cond[stem]
        sums[i] += (1, g, pf, cf, mf, pc, cc, mc)

    def metrics(t):  # t: (..., 8) totals
        img, gt = t[..., 0], np.maximum(t[..., 1], 1)
        return {
            "recall_centre": (t[..., 6] - t[..., 3]) / gt,
            "recall_iou50": (t[..., 7] - t[..., 4]) / gt,
            "boxes_per_image": (t[..., 5] - t[..., 2]) / np.maximum(img, 1),
        }

    point_totals = sums.sum(axis=0)
    point = metrics(point_totals)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(videos), [1.0 / len(videos)] * len(videos),
                              size=n_resamples)
    boot = metrics(weights @ sums)
    img, gt = point_totals[0], max(point_totals[1], 1)
    out = {
        "images": int(img), "videos": len(videos),
        "full": {"recall_centre": point_totals[3] / gt, "recall_iou50": point_totals[4] / gt,
                 "boxes_per_image": point_totals[2] / img},
        "cond": {"recall_centre": point_totals[6] / gt, "recall_iou50": point_totals[7] / gt,
                 "boxes_per_image": point_totals[5] / img},
        "delta": {},
    }
    for key, value in point.items():
        samples = boot[key]
        out["delta"][key] = {
            "point": round(float(value), 5),
            "ci95": [round(float(np.percentile(samples, 2.5)), 5),
                     round(float(np.percentile(samples, 97.5)), 5)],
            "frac_resamples_le_0": round(float((samples <= 0).mean()), 4),
        }
    for side in ("full", "cond"):
        out[side] = {k: round(float(v), 5) for k, v in out[side].items()}
    return out


def text_identity(left_dir, right_dir):
    """Stems in common, and how many of them have byte-identical raw text."""
    left = os.path.join(left_dir, "results_raw")
    right = os.path.join(right_dir, "results_raw")
    common = sorted(set(os.listdir(left)) & set(os.listdir(right)))
    same = sum(open(os.path.join(left, f), "rb").read() ==
               open(os.path.join(right, f), "rb").read() for f in common)
    return len(common), same


def report(args):
    present = [c for c in CONDITIONS
               if os.path.isdir(os.path.join(args.runs, c, "results"))]
    if "full" not in present:
        raise SystemExit("no full run under {} -- nothing to compare against".format(
            args.runs))
    prompt_report = os.path.join(HERE, "prompt_report.py")
    json_path = os.path.join(args.runs, "prompt_report.json")
    md_path = os.path.join(args.runs, "prompt_report.md")
    command = [sys.executable, prompt_report, "--runs", args.runs, "--root", args.root,
               "--variants", ",".join(present), "--save-json", json_path,
               "--markdown", md_path]
    if args.rescore:
        command.append("--rescore")
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)

    diffs = {}
    for condition in present[1:]:
        result = subprocess.run(
            [sys.executable, prompt_report, "--diff",
             os.path.join(args.runs, "full"), os.path.join(args.runs, condition)],
            check=True, capture_output=True, text=True)
        diffs[condition] = result.stdout
        print(result.stdout)

    class_names = read_prompt_classes(args.root)
    matches = {c: per_image_matches(os.path.join(args.runs, c), args, class_names)
               for c in present}
    with open(json_path, encoding="utf-8") as handle:
        table = json.load(handle)

    result = {"runs": args.runs, "conditions": present, "bootstrap": {},
              "text_identical_vs_full": {}, "prompt_report": table, "diffs": diffs,
              "n_resamples": args.n_resamples, "seed": args.seed}
    for condition in present[1:]:
        result["bootstrap"][condition] = paired_bootstrap(
            matches["full"], matches[condition], args.n_resamples, args.seed)
        n, same = text_identity(os.path.join(args.runs, "full"),
                                os.path.join(args.runs, condition))
        result["text_identical_vs_full"][condition] = {"common": n, "identical": same}

    lines = ["# VFE ablation", "",
             "Test split, P0_baseline prompt, one model per condition via hooks. Deltas "
             "are condition - full with 95% CIs from a paired bootstrap over the {} test "
             "videos ({} resamples).".format(
                 next(iter(result["bootstrap"].values()))["videos"]
                 if result["bootstrap"] else "?", args.n_resamples), "",
             "| condition | text = full | box/img | R@ctr | R@.5 | Δ box/img [95% CI] | "
             "Δ R@ctr [95% CI] | Δ R@.5 [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    fb = matches["full"]
    full_gt = max(sum(v[0] for v in fb.values()), 1)
    lines.append("| full | - | {:.3f} | {:.4f} | {:.4f} | | | |".format(
        sum(v[1] for v in fb.values()) / max(len(fb), 1),
        sum(v[2] for v in fb.values()) / full_gt, sum(v[3] for v in fb.values()) / full_gt))

    def fmt(d, digits):
        return "{:+.{p}f} [{:+.{p}f}, {:+.{p}f}]".format(d["point"], *d["ci95"], p=digits)

    for condition in present[1:]:
        b = result["bootstrap"][condition]
        t = result["text_identical_vs_full"][condition]
        lines.append("| {} | {}/{} | {:.3f} | {:.4f} | {:.4f} | {} | {} | {} |".format(
            condition, t["identical"], t["common"], b["cond"]["boxes_per_image"],
            b["cond"]["recall_centre"], b["cond"]["recall_iou50"],
            fmt(b["delta"]["boxes_per_image"], 3), fmt(b["delta"]["recall_centre"], 4),
            fmt(b["delta"]["recall_iou50"], 4)))
    lines += ["", "## prompt_report.py", "", open(md_path, encoding="utf-8").read(), "",
              "## prompt_report.py --diff full <condition>", ""]
    for condition, text in diffs.items():
        lines += ["### {}".format(condition), "", "```", text.rstrip(), "```", ""]

    with open(os.path.join(args.runs, "ablation_report.json"), "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2)
    with open(os.path.join(args.runs, "ablation_report.md"), "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(present) + 6]))
    print("wrote {}/ablation_report.{{json,md}}".format(args.runs))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="0", help="CUDA device index.")

    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--render", type=int, default=0,
                        help="--measure: write vfe.png for the first N frames.")

    parser.add_argument("--conditions", default="full",
                        help="Comma-separated: " + ",".join(CONDITIONS))
    parser.add_argument("--subsample", action="store_true",
                        help="Use the pinned screening subsample instead of the full "
                             "split (determinism gates).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="",
                        help="Suffix for the run dir, e.g. _cuda1 for a gate run.")
    parser.add_argument("--hook-noop", action="store_true",
                        help="Pass-through hooks and wrappers (determinism gate).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--query-max-pixels", type=int, default=1338)
    parser.add_argument("--exemplar-max-pixels", type=int, default=256)

    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rescore", action="store_true",
                        help="--report: re-run the scorers even if their json exists.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.self_test:
        return self_test(args)
    if args.measure:
        return measure(args)
    if args.report:
        return report(args)
    return run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
