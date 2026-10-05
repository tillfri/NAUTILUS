"""Did the VFE train at all? Weight drift of ``visual.nautilus_*`` against its init.

``vfe_ablation.py`` showed that switching the VFE off at inference changes nothing.
The open caveat is training-time: "a model trained *without* the VFE might score
differently". That caveat only has teeth if the VFE parameters actually moved during
training. This script measures how far they moved, on CPU, straight out of the
safetensors shards. Nothing is built and nothing runs on a GPU.

What the code says the init is (``Qwen2_5_VL_Nautilus_ForConditionalGeneration.py``,
``Nautilus_layers.py``, ``train_qwen.py``; transformers 4.51.3, peft 0.12):

  1. ``from_pretrained(Qwen/Qwen2.5-VL-7B-Instruct)`` finds no ``visual.nautilus_*``
     keys and applies the *top-level* ``_init_weights`` to them: Linear and LayerNorm
     weights normal(0, initializer_range=0.02), every bias zeroed.
  2. ``get_peft_model`` puts the VFE into ``modules_to_save``. ``init_nautilus_model``
     falls into its ``except`` branch and re-runs ``MLP.weight_init`` /
     ``CrossAttentionNetwork.linear_init`` / ``GlobalQueries.weight_init`` on the
     ``modules_to_save.default`` copy, i.e. the copy that is trained and saved:
     fc1/fc2, ``in_proj_weight``, ``out_proj.weight`` and ``output_linear.weight`` get
     kaiming_normal(fan_in, relu), every bias gets zeros, ``global_queries`` gets
     xavier_uniform. The LayerNorm keeps the step-1 normal(0, 0.02) weight and zero bias.

  So **every bias in the VFE starts at exactly 0**. That includes
  ``nn.MultiheadAttention``'s ``in_proj_bias`` and ``out_proj.bias``, which its own
  ``_reset_parameters`` and ``linear_init`` both zero. A bias's saved value *is* its
  drift, and near 0 bf16 resolves it finely. This is the clean observable (section 1).

Sections of ``summary.json`` / ``report.md``:

  1  zero-init tensors: max|b|, mean|b|, RMS and the share still exactly 0.
     ``in_proj_bias`` is split into its q/k/v thirds. The query is a single token
     per image, so the **key bias adds the same constant to every attention logit and
     its true gradient is exactly 0**. Its drift is what Adam does with numerical noise
     alone, measured inside the same tensor and the same optimizer group.
  2  random-init tensors: the checkpoint against fresh draws from the same init
     function on a **CPU bf16** tensor, ``--seeds`` of them. The checkpoint's
     max|w| sits on the CPU-bf16 Box-Muller ceiling, which identifies where the init
     ran. Reported: std and max|w| next to the seed spread, and the two-sample KS
     statistic (checkpoint vs fresh) next to fresh vs fresh. The original seed is
     unknown, so "fraction of weights within one ULP of their never-moved value" is
     not computable directly. What is computable: a CPU-bf16 init only produces a
     finite *lattice* of values, so the share of checkpoint values off that lattice
     is a lower bound on the share that moved. It is split by |w|, because drift
     smaller than the local bf16 ULP rounds back onto the init value. A held-out
     fresh draw calibrates the lattice's completeness.
  3  the drift a trained parameter could have reached. The logged lr curve
     (``logs/qwen_logs.json``) is scaled by ``nautilus_lr / lr`` from the launcher.
     Sum lr_t bounds a directed Adam walk, (1-b1)/sqrt(1-b2) * sum lr_t is Adam's hard
     per-step bound, and sqrt(sum lr_t^2) is the scale of an undirected one.
  4  positive control: ``visual.merger.*``, ``embed_tokens`` and ``lm_head`` against
     the stock checkpoint. These were trained in ``modules_to_save`` at lr, 100x the
     VFE's. This assumes the stock copy (HF cache, pulled 25 Aug) is the revision the
     authors started from.
  5  negative controls, which must be bit-identical: the Qwen ViT
     (``tune_mm_vision False``), the frozen LLM norms and q/k/v biases (LoRA, bias
     'none') against stock, and ``visual.nautilus_encoder.*`` against the DINOv2
     source weights cast to bf16. A failure breaks the method or its assumptions, and
     the run says so loudly. Every other common key (LoRA-merged LLM linears) is
     censused as changed/unchanged.

Usage (container, CPU only, ~5 min, reads ~33 GB of shards):

    docker exec nautilus-qwen bash -lc \\
      "cd /workspace/NAUTILUS/qwen-vl-finetune/scripts && \\
       python3 vfe_weight_drift.py --out /workspace/runs/vfe_weight_drift"
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.append(str(PROJECT_ROOT))

import torch  # noqa: E402

from dino_scramble_experiment import read_checkpoint_tensors  # noqa: E402

DEFAULT_CHECKPOINT = "/workspace/weights/qwen-instruct-7b-weights"
DEFAULT_STOCK = "/workspace/weights/qwen2.5-vl-7b-instruct"
DEFAULT_DINO = "/workspace/weights/dinov2-weights/dinov2.pth"
DEFAULT_LOG = "/workspace/NAUTILUS/logs/qwen_logs.json"
DEFAULT_LAUNCHER = str(SCRIPTS_DIR / "nautilus_finetune" / "nautilus_lora_sft_7b_lora.sh")

# HF TrainingArguments defaults; the launcher overrides none of them.
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

ZERO_INIT = [
    "nautilus_w1_mlp.ln.bias",
    "nautilus_w1_mlp.fc1.bias",
    "nautilus_w1_mlp.fc2.bias",
    "nautilus_dark_mlp.ln.bias",
    "nautilus_dark_mlp.fc1.bias",
    "nautilus_dark_mlp.fc2.bias",
    "nautilus_dark_attn.cross_attention.in_proj_bias",
    "nautilus_dark_attn.cross_attention.out_proj.bias",
    "nautilus_dark_attn.output_linear.bias",
]

KAIMING = [
    "nautilus_w1_mlp.fc1.weight",
    "nautilus_w1_mlp.fc2.weight",
    "nautilus_dark_mlp.fc1.weight",
    "nautilus_dark_mlp.fc2.weight",
    "nautilus_dark_attn.cross_attention.in_proj_weight",
    "nautilus_dark_attn.cross_attention.out_proj.weight",
    "nautilus_dark_attn.output_linear.weight",
]
LN_WEIGHTS = ["nautilus_w1_mlp.ln.weight", "nautilus_dark_mlp.ln.weight"]
XAVIER = ["nautilus_global_queries.param"]


# ------------------------------------------------------------------------ helpers


class Shards:
    """Lazy key -> tensor access over a sharded safetensors checkpoint, raw dtype."""

    def __init__(self, checkpoint_dir):
        from safetensors import safe_open

        self.dir = Path(checkpoint_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def keys(self):
        return self.weight_map.keys()

    def get(self, key):
        shard = self.weight_map[key]
        if shard not in self._open:
            self._open[shard] = self._safe_open(str(self.dir / shard), framework="pt")
        return self._open[shard].get_tensor(key)


def bf16_ulp(x):
    """Spacing of bf16 values at magnitude |x| (8 significant bits)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def ks_2samp(a, b):
    """Two-sample KS statistic; exact on tied (bf16-quantised) values."""
    a = a.flatten().double().sort().values
    b = b.flatten().double().sort().values
    grid = torch.cat([a, b]).unique()
    cdf_a = torch.searchsorted(a, grid, right=True).double() / a.numel()
    cdf_b = torch.searchsorted(b, grid, right=True).double() / b.numel()
    return float((cdf_a - cdf_b).abs().max())


def drift_stats(x):
    x = x.float()
    return {
        "n": x.numel(),
        "max_abs": float(x.abs().max()),
        "mean_abs": float(x.abs().mean()),
        "rms": float(x.pow(2).mean().sqrt()),
        "frac_exactly_zero": float((x == 0).double().mean()),
    }


def diff_stats(new, old, chunk_rows=8192):
    """||new-old|| / ||old||, max|d|, RMS d and share changed, chunked over dim 0."""
    if new.shape != old.shape:
        raise SystemExit(f"shape mismatch {tuple(new.shape)} vs {tuple(old.shape)}")
    sq_d = sq_o = 0.0
    max_d = 0.0
    changed = 0
    n = new.numel()
    new2 = new.reshape(new.shape[0], -1) if new.dim() else new.reshape(1, 1)
    old2 = old.reshape(old.shape[0], -1) if old.dim() else old.reshape(1, 1)
    for i in range(0, new2.shape[0], chunk_rows):
        a = new2[i:i + chunk_rows].double()
        b = old2[i:i + chunk_rows].double()
        d = a - b
        sq_d += float(d.pow(2).sum())
        sq_o += float(b.pow(2).sum())
        max_d = max(max_d, float(d.abs().max()))
        changed += int((d != 0).sum())
    return {
        "shape": list(new.shape),
        "rel_norm": math.sqrt(sq_d / sq_o) if sq_o else float("nan"),
        "max_abs_delta": max_d,
        "rms_delta": math.sqrt(sq_d / n),
        "frac_changed": changed / n,
    }


def summarise(values):
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean()), "sd": float(t.std()) if len(values) > 1 else 0.0,
            "min": float(t.min()), "max": float(t.max())}


# ------------------------------------------------------------------- sections


def zero_init_section(vfe):
    out = {}
    for name in ZERO_INIT:
        t = vfe[name]
        if name.endswith("in_proj_bias"):
            for part, s in zip("qkv", t.chunk(3)):
                entry = drift_stats(s)
                entry["zero_true_gradient"] = part == "k"
                out[f"{name}[{part}]"] = entry
        else:
            out[name] = drift_stats(t)
    return out


def fresh_draw(name, shape, initializer_range):
    w = torch.empty(shape, dtype=torch.bfloat16)  # CPU, see section 2 in the docstring
    if name in KAIMING:
        torch.nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
    elif name in LN_WEIGHTS:
        w.normal_(mean=0.0, std=initializer_range)
    else:
        torch.nn.init.xavier_uniform_(w)
    return w.float()


def init_scale(name, shape, initializer_range):
    if name in KAIMING:
        return {"kind": "kaiming_normal(fan_in, relu)", "std": math.sqrt(2.0 / shape[1])}
    if name in LN_WEIGHTS:
        return {"kind": "normal(0, initializer_range)", "std": initializer_range}
    fan_in = shape[1] * math.prod(shape[2:])
    fan_out = shape[0] * math.prod(shape[2:])
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return {"kind": "xavier_uniform", "bound": bound, "std": bound / math.sqrt(3.0)}


LATTICE_BINS = [0.0, 1e-3, 4e-3, 1.6e-2, math.inf]


def lattice_draw(name, shape, initializer_range, n=1 << 22):
    """A large CPU-bf16 draw with the same init parameters as ``name``.

    CPU bf16 ``normal_``/``uniform_`` work from a quantised uniform, so a fresh init
    only ever produces a fixed, finite set of bf16 values. Drawing many large tensors
    approximates that set.
    """
    if name in KAIMING:
        w = torch.empty(n // shape[1], shape[1], dtype=torch.bfloat16)  # fan_in = shape[1]
        torch.nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
    elif name in LN_WEIGHTS:
        w = torch.empty(n, dtype=torch.bfloat16).normal_(mean=0.0, std=initializer_range)
    else:
        w = torch.empty(n, dtype=torch.bfloat16)
        bound = init_scale(name, shape, initializer_range)["bound"]
        w.uniform_(-bound, bound)  # what xavier_uniform_ calls
    return w.flatten().unique()


def off_lattice(name, ckpt, initializer_range, seeds):
    """Share of checkpoint values a fresh init cannot produce, per |w| bin.

    An element off the lattice has certainly moved; one on it may still have moved
    onto another lattice value, so this is a lower bound on the share that moved. A
    held-out fresh draw (seed ``seeds``, not in the lattice) calibrates how complete
    the lattice is: its off-lattice share should be ~0.
    """
    shape = tuple(ckpt.shape)
    lattice = torch.tensor([], dtype=torch.bfloat16)
    growth = []
    for seed in range(seeds):
        torch.manual_seed(1000 + seed)
        before = lattice.numel()
        lattice = torch.cat([lattice, lattice_draw(name, shape, initializer_range)]).unique()
        growth.append(lattice.numel() - before)
    torch.manual_seed(seeds)
    held_out = fresh_draw(name, shape, initializer_range).to(torch.bfloat16).flatten()
    ckpt_bf16 = ckpt.to(torch.bfloat16).flatten()
    lattice_bits = lattice.view(torch.int16)  # isin has no bf16 kernel; bits are exact

    def frac_off(x):
        return float((~torch.isin(x.view(torch.int16), lattice_bits)).double().mean())

    bins = []
    for lo, hi in zip(LATTICE_BINS[:-1], LATTICE_BINS[1:]):
        entry = {"abs_lo": lo, "abs_hi": hi if math.isfinite(hi) else None,
                 "bf16_ulp_at_lo": bf16_ulp(max(lo, 1e-4))}
        for label, x in (("ckpt", ckpt_bf16), ("held_out_fresh", held_out)):
            a = x.float().abs()
            sel = x[(a >= lo) & (a < hi)]
            entry[f"{label}_n"] = sel.numel()
            entry[f"{label}_frac_off"] = frac_off(sel) if sel.numel() else None
        bins.append(entry)
    return {
        "lattice_size": lattice.numel(),
        "lattice_growth_last_seed": growth[-1],
        "distinct_values_ckpt": ckpt_bf16.unique().numel(),
        "distinct_values_held_out": held_out.unique().numel(),
        "ckpt_frac_off": frac_off(ckpt_bf16),
        "held_out_frac_off": frac_off(held_out),
        "bins": bins,
    }


def random_init_section(vfe, initializer_range, seeds):
    out = {}
    for name in KAIMING + LN_WEIGHTS + XAVIER:
        ckpt = vfe[name]
        shape = tuple(ckpt.shape)
        scale = init_scale(name, shape, initializer_range)
        stds, maxs, ks_ckpt, ks_fresh = [], [], [], []
        prev = None
        for seed in range(seeds):
            torch.manual_seed(seed)
            fresh = fresh_draw(name, shape, initializer_range)
            stds.append(float(fresh.std()))
            maxs.append(float(fresh.abs().max()))
            ks_ckpt.append(ks_2samp(ckpt, fresh))
            if prev is not None:
                ks_fresh.append(ks_2samp(prev, fresh))
            prev = fresh
        std = float(ckpt.std())
        std_seed = summarise(stds)
        out[name] = {
            "shape": list(shape),
            "init": scale,
            "mean": float(ckpt.mean()),
            "std": std,
            "std_over_init_std": std / scale["std"],
            "max_abs": float(ckpt.abs().max()),
            "fresh_std": std_seed,
            "std_z_vs_fresh": ((std - std_seed["mean"]) / std_seed["sd"]
                               if std_seed["sd"] else None),
            "fresh_max_abs": summarise(maxs),
            "ks_ckpt_vs_fresh": summarise(ks_ckpt),
            "ks_fresh_vs_fresh": summarise(ks_fresh),
            "bf16_ulp_at_median_abs": bf16_ulp(float(ckpt.abs().median())),
            "off_init_lattice": off_lattice(name, ckpt, initializer_range, seeds),
        }
    return out


def lr_section(log_path, launcher_path):
    launcher = Path(launcher_path).read_text()

    def launcher_value(pattern):
        m = re.search(pattern, launcher, re.M)
        return float(m.group(1)) if m else None

    lr = launcher_value(r"^lr=(\S+)")
    nautilus_lr = launcher_value(r"^nautilus_lr=(\S+)")
    script_epochs = launcher_value(r"--num_train_epochs (\S+)")

    log = json.loads(Path(log_path).read_text())
    history = log["log_history"]
    lrs = [e["learning_rate"] for e in history if "learning_rate" in e]
    steps = [e["step"] for e in history if "learning_rate" in e]
    ratio = nautilus_lr / lr
    sum_lr = sum(lrs)
    rw = math.sqrt(sum(v * v for v in lrs))
    adam_hard = (1 - ADAM_BETA1) / math.sqrt(1 - ADAM_BETA2)
    return {
        "launcher": {"lr": lr, "nautilus_lr": nautilus_lr, "num_train_epochs": script_epochs},
        "log": {
            "entries_with_lr": len(lrs),
            "steps_contiguous_from_1": steps == list(range(1, len(steps) + 1)),
            "peak_lr": max(lrs),
            "max_steps": log.get("max_steps"),
            "global_step": log.get("global_step"),
            "num_train_epochs": log.get("num_train_epochs"),
            "final_epoch": log.get("epoch"),
        },
        "mismatches": {
            "epochs_log_vs_launcher": [log.get("num_train_epochs"), script_epochs],
            "global_step_vs_max_steps": [log.get("global_step"), log.get("max_steps")],
        },
        "adam": {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "eps": ADAM_EPS,
                 "hard_step_factor": adam_hard},
        "main_group": {"sum_lr": sum_lr, "sqrt_sum_lr2": rw},
        "nautilus_group": {
            "lr_ratio": ratio,
            "directed_bound_sum_lr": sum_lr * ratio,
            "hard_bound": adam_hard * sum_lr * ratio,
            "random_walk_scale": rw * ratio,
        },
    }


def classify(key):
    if key.startswith("visual.nautilus_encoder."):
        return "dino_encoder"
    if key.startswith("visual.nautilus_"):
        return "vfe"
    if key.startswith("visual.merger."):
        return "merger"
    if key.startswith("visual."):
        return "qwen_vit"
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return key.split(".")[-2]
    if key == "model.norm.weight" or key.endswith("layernorm.weight"):
        return "llm_norm"
    if key.endswith("_proj.bias"):
        return "llm_proj_bias"
    if key.endswith("_proj.weight"):
        return "llm_lora_linear"
    return "other"


def census_section(naut, stock, dino_path):
    stock_keys = set(stock.keys())
    groups = defaultdict(lambda: {"tensors": 0, "identical": 0, "max_abs_delta": 0.0,
                                  "examples_changed": []})
    positive = {}
    for key in sorted(naut.keys()):
        group = classify(key)
        if group in ("vfe", "dino_encoder"):
            continue
        if key not in stock_keys:
            groups["only_in_nautilus"]["tensors"] += 1
            continue
        new, old = naut.get(key), stock.get(key)
        g = groups[group]
        g["tensors"] += 1
        if new.dtype == old.dtype and new.shape == old.shape and torch.equal(new, old):
            g["identical"] += 1
        else:
            d = diff_stats(new, old)
            g["max_abs_delta"] = max(g["max_abs_delta"], d["max_abs_delta"])
            if len(g["examples_changed"]) < 3:
                g["examples_changed"].append(key)
            if group in ("merger", "embed_tokens", "lm_head"):
                positive[key] = d
        if group in ("merger", "embed_tokens", "lm_head") and key not in positive:
            positive[key] = diff_stats(new, old)

    dino_state = torch.load(dino_path, map_location="cpu", weights_only=True)
    enc_keys = [k for k in naut.keys() if k.startswith("visual.nautilus_encoder.")]
    dino = {"tensors": len(enc_keys), "identical": 0, "missing_in_source": 0,
            "max_abs_delta": 0.0, "examples_changed": []}
    for key in enc_keys:
        src = dino_state.get(key[len("visual.nautilus_encoder."):])
        if src is None:
            dino["missing_in_source"] += 1
            continue
        new = naut.get(key)
        ref = src.to(new.dtype)
        if new.shape == ref.shape and torch.equal(new, ref):
            dino["identical"] += 1
        else:
            dino["max_abs_delta"] = max(dino["max_abs_delta"],
                                        float((new.float() - ref.float()).abs().max()))
            if len(dino["examples_changed"]) < 3:
                dino["examples_changed"].append(key)
    dino["source_tensors"] = len(dino_state)

    groups = {k: dict(v) for k, v in groups.items()}
    negatives = {
        "qwen_vit": groups.get("qwen_vit"),
        "llm_norm": groups.get("llm_norm"),
        "llm_proj_bias": groups.get("llm_proj_bias"),
        "dino_encoder_vs_source": dino,
    }
    passed = all(v is not None and v["identical"] == v["tensors"]
                 for v in negatives.values())
    return {"groups": groups, "positive": positive, "negatives": negatives,
            "negative_controls_pass": passed}


# --------------------------------------------------------------------- report


def fmt(x, spec=".3e"):
    return "–" if x is None else format(x, spec)


def write_report(summary, path):
    lr = summary["lr"]
    ng = lr["nautilus_group"]
    lines = ["# VFE weight drift", ""]
    lines += ["## 1. Zero-init tensors (value = drift)", "",
              "| tensor | n | max\\|b\\| | mean\\|b\\| | RMS | exactly 0 | max/directed bound | RMS/random walk |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in summary["zero_init"].items():
        tag = " **(true grad ≡ 0)**" if e.get("zero_true_gradient") else ""
        lines.append(
            f"| `{name}`{tag} | {e['n']} | {fmt(e['max_abs'])} | {fmt(e['mean_abs'])} | "
            f"{fmt(e['rms'])} | {e['frac_exactly_zero']:.3f} | "
            f"{e['max_abs'] / ng['directed_bound_sum_lr']:.4f} | "
            f"{e['rms'] / ng['random_walk_scale']:.2f} |")
    lines += ["", "## 2. Random-init tensors vs fresh CPU-bf16 draws "
              f"({summary['args']['seeds']} seeds)", "",
              "| tensor | init | std / init std | std z vs fresh | max\\|w\\| | fresh max\\|w\\| (min–max) | KS ckpt–fresh (mean) | KS fresh–fresh (mean, max) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, e in summary["random_init"].items():
        lines.append(
            f"| `{name}` | {e['init']['kind']} | {e['std_over_init_std']:.4f} | "
            f"{fmt(e['std_z_vs_fresh'], '+.2f')} | {e['max_abs']:.4f} | "
            f"{e['fresh_max_abs']['min']:.4f}–{e['fresh_max_abs']['max']:.4f} | "
            f"{e['ks_ckpt_vs_fresh']['mean']:.4f} | "
            f"{e['ks_fresh_vs_fresh']['mean']:.4f}, {e['ks_fresh_vs_fresh']['max']:.4f} |")
    lines += ["", "### Off the init lattice (values a fresh CPU-bf16 init cannot produce)", "",
              "Share off-lattice, checkpoint / held-out fresh draw, per |w| bin. "
              "Off-lattice = certainly moved (a lower bound).", "",
              "| tensor | lattice size (+last seed) | distinct ckpt / fresh | all | "
              + " | ".join(f"\\|w\\| ∈ [{lo:g}, {hi:g})" for lo, hi in
                           zip(LATTICE_BINS[:-1], LATTICE_BINS[1:])) + " |",
              "|---|---:|---:|---:|" + "---:|" * (len(LATTICE_BINS) - 1)]
    for name, e in summary["random_init"].items():
        o = e["off_init_lattice"]
        cells = []
        for b in o["bins"]:
            cells.append("–" if b["ckpt_frac_off"] is None else
                         f"{b['ckpt_frac_off']:.3f} / {fmt(b['held_out_fresh_frac_off'], '.3f')} "
                         f"(n={b['ckpt_n']})")
        lines.append(
            f"| `{name}` | {o['lattice_size']} (+{o['lattice_growth_last_seed']}) | "
            f"{o['distinct_values_ckpt']} / {o['distinct_values_held_out']} | "
            f"{o['ckpt_frac_off']:.3f} / {o['held_out_frac_off']:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. How far a trained VFE parameter could have moved", "",
              f"- launcher: lr {lr['launcher']['lr']}, nautilus_lr {lr['launcher']['nautilus_lr']} "
              f"(ratio {ng['lr_ratio']}), num_train_epochs {lr['launcher']['num_train_epochs']}",
              f"- log: {lr['log']['entries_with_lr']} steps (contiguous: "
              f"{lr['log']['steps_contiguous_from_1']}), peak lr {lr['log']['peak_lr']}, "
              f"max_steps {lr['log']['max_steps']}, global_step {lr['log']['global_step']}, "
              f"num_train_epochs {lr['log']['num_train_epochs']}",
              f"- nautilus group, directed bound Σ lr_t: **{ng['directed_bound_sum_lr']:.3e}**",
              f"- Adam hard bound ({lr['adam']['hard_step_factor']:.2f} × Σ lr_t): {ng['hard_bound']:.3e}",
              f"- undirected random-walk scale sqrt(Σ lr_t²): **{ng['random_walk_scale']:.3e}**",
              f"- init scale for comparison: initializer_range {summary['initializer_range']}; "
              f"bf16 ULP at 0.02 = {bf16_ulp(0.02):.3e}"]
    lines += ["", "## 4. Positive control: trained at lr, vs stock", "",
              "| tensor | shape | ‖Δ‖/‖θ₀‖ | max\\|Δ\\| | RMS Δ | changed | RMS Δ × lr ratio |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key, d in summary["census"]["positive"].items():
        lines.append(
            f"| `{key}` | {d['shape']} | {d['rel_norm']:.3e} | {d['max_abs_delta']:.3e} | "
            f"{d['rms_delta']:.3e} | {d['frac_changed']:.3f} | "
            f"{d['rms_delta'] * ng['lr_ratio']:.3e} |")
    lines += ["", "## 5. Negative controls (must be bit-identical) and census", "",
              f"**negative controls pass: {summary['census']['negative_controls_pass']}**", "",
              "| group | tensors | identical | max\\|Δ\\| | changed examples |",
              "|---|---:|---:|---:|---|"]
    rows = dict(summary["census"]["groups"])
    rows["dino_encoder_vs_source"] = summary["census"]["negatives"]["dino_encoder_vs_source"]
    for group, g in rows.items():
        lines.append(f"| {group} | {g['tensors']} | {g['identical']} | "
                     f"{fmt(g['max_abs_delta'])} | {', '.join(g['examples_changed'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--stock", default=DEFAULT_STOCK)
    p.add_argument("--dino", default=DEFAULT_DINO)
    p.add_argument("--log", default=DEFAULT_LOG)
    p.add_argument("--launcher", default=DEFAULT_LAUNCHER)
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))

    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    initializer_range = config["initializer_range"]
    vfe = read_checkpoint_tensors(args.checkpoint, "visual.")
    vfe = {k: v for k, v in vfe.items() if k.startswith("nautilus_")
           and not k.startswith("nautilus_encoder.")}
    expected = set(ZERO_INIT + KAIMING + LN_WEIGHTS + XAVIER)
    if set(vfe) != expected:
        raise SystemExit(f"VFE key set differs from the code's: {sorted(set(vfe) ^ expected)}")

    print("[1/5] zero-init tensors", flush=True)
    zero_init = zero_init_section(vfe)
    print("[2/5] random-init tensors vs fresh draws", flush=True)
    random_init = random_init_section(vfe, initializer_range, args.seeds)
    print("[3/5] lr bound", flush=True)
    lr = lr_section(args.log, args.launcher)
    print("[4-5/5] census vs stock + DINOv2 source", flush=True)
    census = census_section(Shards(args.checkpoint), Shards(args.stock), args.dino)

    summary = {
        "args": vars(args),
        "initializer_range": initializer_range,
        "torch": torch.__version__,
        "zero_init": zero_init,
        "random_init": random_init,
        "lr": lr,
        "census": census,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, out / "report.md")
    print(f"wrote {out / 'summary.json'} and {out / 'report.md'}")
    if not census["negative_controls_pass"]:
        print("!!! NEGATIVE CONTROL FAILED -- a tensor that should be bit-identical "
              "differs; the drift numbers cannot be trusted until this is explained. "
              "See section 5 of report.md.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
