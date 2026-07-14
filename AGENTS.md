# AGENTS.md — NAUTILUS-llava

Guidance for AI agents working on this repository.

---

## Agent Execution Policy

- **All Python code runs inside a Docker container.** Do not attempt to create virtual environments, install packages, or execute Python scripts directly. Assume all required dependencies (PyTorch, transformers, matplotlib, Pillow, etc.) are already installed in the container environment.
- **The user runs all code.** When Python scripts are needed, write or edit the file and instruct the user to execute it. Never invoke `python`, `uv run`, `pip`, or `conda` commands yourself.

---

## What This Repo Is

**NAUTILUS** is a Large Multimodal Model (LMM) for underwater scene understanding (NeurIPS 2025). It addresses the domain gap where standard vision-language models fail on underwater imagery due to light absorption, backscatter, and color distortion.

### Two model variants live side-by-side

| Variant | Directory | Backbone |
|---|---|---|
| NAUTILUS (LLaVA-1.5) | `LLaVA/` | LLaVA-1.5 + Vicuna-7B |
| NAUTILUS (Qwen2.5-VL) | `qwen-vl-finetune/` | Qwen2.5-VL-7B-Instruct |

Both share the same core architectural innovation: a **Vision Feature Enhancement (VFE)** module that uses a physical underwater imaging model to restore degraded visual features.

### Supported tasks (8 total)

Image Classification, Image Captioning, Region Captioning, VQA, Grounding, Object Detection, Object Counting, Region Classification.

---

## Repository Layout

```
NAUTILUS-llava/
├── README.md                        # Install, train, infer, evaluate instructions
├── AGENTS.md                        # This file
├── Figs/                            # Architecture diagrams (Intro.png, pipeline.png)
├── eval/                            # Shared evaluation framework for both variants
│   ├── eval4nautilus_llava.py       # Evaluation entry point (LLaVA)
│   ├── eval4nautilus_qwen.py        # Evaluation entry point (Qwen)
│   ├── eval_nautilus_llava.sh       # Shell launcher (LLaVA)
│   ├── eval_nautilus_qwen.sh        # Shell launcher (Qwen)
│   ├── utils.py                     # Task routing, bbox scaling, token duplication
│   └── Evaluation_pack/             # Per-task metric modules
│       ├── classification.py        # Accuracy, F1, MCC, ROC-AUC
│       ├── count.py                 # MAE, RMSE, choice accuracy
│       ├── detection.py             # COCO mAP, AP@0.5/0.75
│       ├── grounding.py             # mIoU, Precision@0.5, COCO AP
│       └── text.py                  # BLEU, CIDEr, ROUGE-L, METEOR
├── LLaVA/                           # NAUTILUS-LLaVA branch
├── logs/                            # Saved eval results and training logs
│   ├── llava_eval.json
│   ├── llava_logs.json
│   ├── qwen_eval.json
│   └── qwen_logs.json
├── qwen-vl-finetune/                # NAUTILUS-Qwen branch
└── utils/
    └── process_vitl_weight.py       # Extract DINOv2 weights from Depth-Anything-V2
```

### `LLaVA/` branch layout

```
LLaVA/
├── pyproject.toml                        # Package: nautilus_llava v1.0.0
├── llava/
│   ├── constants.py                      # Special token constants
│   ├── conversation.py                   # Conversation templates
│   ├── mm_utils.py                       # Image tokenization utilities
│   ├── model/
│   │   ├── builder.py                    # load_pretrained_model() — main loader
│   │   ├── llava_vfe_arch.py             # *** Core VFE architecture ***
│   │   ├── vfe_layer.py                  # CrossAttentionNetwork, MLP
│   │   ├── dinov2.py                     # DINOv2 ViT-L encoder
│   │   ├── dinov2_layers/                # DINOv2 building blocks
│   │   ├── language_model/
│   │   │   └── vfellava_llama.py         # VFELLaVALlamaForCausalLM
│   │   ├── multimodal_encoder/
│   │   │   └── clip_encoder.py           # CLIP ViT-L/14-336
│   │   └── multimodal_projector/
│   │       └── builder.py                # MLP projector (mlp2x_gelu)
│   ├── train/
│   │   ├── train.py                      # Main training script
│   │   ├── train_mem.py                  # Memory-efficient variant (used for finetune)
│   │   └── llava_trainer.py              # Custom HF Trainer subclass
│   ├── eval/                             # Upstream LLaVA benchmark eval scripts
│   └── serve/                            # Gradio UI, CLI, FastAPI controller
└── scripts/
    ├── nautilus_finetune/
    │   └── finetune_nautilus_lora.sh     # *** Main training launcher ***
    ├── inference/
    │   └── inference.py                  # Single-image inference
    ├── merge_lora_weights.py             # Merge LoRA into base model
    ├── zero2.json / zero3.json / zero3_offload.json  # DeepSpeed configs
    └── v1_5/                             # Standard LLaVA-1.5 scripts (pretrain, finetune, eval)
```

### `qwen-vl-finetune/` branch layout

```
qwen-vl-finetune/
├── requirements.txt                       # Pinned Python dependencies
├── qwenvl/
│   ├── nautilus_model/
│   │   ├── Qwen2_5_VL_Nautilus_ForConditionalGeneration.py  # *** Core VFE architecture ***
│   │   ├── Nautilus_layers.py             # CrossAttentionNetwork, MLP, GlobalQueries
│   │   ├── dinov2.py                      # DINOv2 ViT-L encoder (same as LLaVA branch)
│   │   └── dinov2_layers/                 # DINOv2 building blocks
│   ├── data/
│   │   ├── __init__.py                    # Dataset registry (Nautilus_Instruct)
│   │   ├── data_qwen.py                   # make_supervised_data_module()
│   │   └── rope2d.py                      # 2D RoPE positional encoding
│   └── train/
│       ├── train_qwen.py                  # *** Main training entry point ***
│       ├── trainer.py                     # Custom trainer (data flattening)
│       ├── argument.py                    # Dataclass configs (Model/Data/Lora/Training args)
│       └── utils.py                       # LoRA helpers
├── scripts/
│   ├── nautilus_finetune/
│   │   └── nautilus_lora_sft_7b_lora.sh  # *** Main training launcher ***
│   ├── inference.py                       # Single-image inference
│   ├── merge_lora.py / merge_lora_qwen.sh # LoRA merging
│   └── zero2.json / zero3.json / zero3_offload.json / zero2_grad_clip.json
├── demo/
│   ├── images/                            # Sample PNG images
│   ├── videos/                            # Sample MP4 videos
│   ├── single_images.json                 # Example dataset format
│   └── video.json                         # Example video dataset format
└── tools/
    ├── check_image.py                     # Validate dataset image paths
    └── process_bbox.ipynb                 # Convert bbox annotations to Qwen format
```

---

## Core Technical Concept: The VFE Module

The VFE module is the central innovation. It runs in two steps motivated by the physical underwater imaging model.

**Step 1 — Backscatter Removal**
- Identifies the darkest image patch (minimum mean brightness)
- Uses a `CrossAttentionNetwork` with a learnable global query against image features to estimate the global backscatter component
- Subtracts it: `filtered_embeds = image_embeds - dark_embeds`

**Step 2 — Light Absorption Restoration**
- DINOv2 ViT-L (depth encoder) extracts features from layer 23
- An MLP maps DINOv2 features to a multiplicative weight: `weight = 1 / exp(-MLP(dino_feat))`
- Multiplies cleaned features: `enhanced = weight * filtered_embeds`

**Output differences between variants:**
- **LLaVA**: original and enhanced features are both projected and concatenated along the sequence dimension
- **Qwen**: ViT output is stacked as `[original, enhanced]` per token, producing 2x token count — this is why `double_image_tokens()` in `eval/utils.py` must be called during Qwen inference

---

## Key Files to Know

### Architecture (read these first for any model change)

| File | Role |
|---|---|
| `LLaVA/llava/model/llava_vfe_arch.py` | VFE-enhanced LLaVA meta-model; contains `ehance_encode_images()` |
| `LLaVA/llava/model/vfe_layer.py` | `CrossAttentionNetwork` and `MLP` for LLaVA |
| `LLaVA/llava/model/language_model/vfellava_llama.py` | `VFELLaVALlamaForCausalLM` — top-level model class |
| `qwen-vl-finetune/qwenvl/nautilus_model/Qwen2_5_VL_Nautilus_ForConditionalGeneration.py` | VFE-enhanced Qwen model; contains `ehance_embeds()` and `restore_image_from_patches()` |
| `qwen-vl-finetune/qwenvl/nautilus_model/Nautilus_layers.py` | `CrossAttentionNetwork`, `MLP`, `GlobalQueries` for Qwen |
| `LLaVA/llava/model/dinov2.py` / `qwen-vl-finetune/qwenvl/nautilus_model/dinov2.py` | DINOv2 ViT-L implementation (identical in both branches) |

### Evaluation

| File | Role |
|---|---|
| `eval/utils.py` | `sortbyid()` task routing, `double_image_tokens()`, `scale_bboxes_in_text()` |
| `eval/eval4nautilus_llava.py` | Full eval pipeline for LLaVA |
| `eval/eval4nautilus_qwen.py` | Full eval pipeline for Qwen |
| `eval/Evaluation_pack/*.py` | Task-specific metric computation |

### Training configuration

| File | Role |
|---|---|
| `LLaVA/scripts/nautilus_finetune/finetune_nautilus_lora.sh` | LLaVA training launcher |
| `qwen-vl-finetune/scripts/nautilus_finetune/nautilus_lora_sft_7b_lora.sh` | Qwen training launcher |
| `qwen-vl-finetune/qwenvl/train/argument.py` | All training argument dataclasses |
| `qwen-vl-finetune/qwenvl/data/__init__.py` | Dataset registry (`Nautilus_Instruct` paths) |

---

## Environment Setup

There are **two separate conda environments** — one per variant. Do not mix them.

### NAUTILUS-LLaVA (CUDA 12.1, Python 3.10)
```bash
conda create -n nautilus_llava python=3.10 -y
conda activate nautilus_llava
cd LLaVA
pip install -e .
pip install flash-attn==2.6.3 --no-build-isolation
```

Key pinned versions (from `LLaVA/pyproject.toml`): `torch==2.1.2`, `transformers==4.45.2`, `peft==0.15.2`, `deepspeed==0.12.6`, `timm==0.6.13`

### NAUTILUS-Qwen (CUDA 12.4, Python 3.10)
```bash
conda create -n nautilus_qwen python=3.10 -y
conda activate nautilus_qwen
cd qwen-vl-finetune
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

Key pinned versions (from `qwen-vl-finetune/requirements.txt`): `torch==2.5.1`, `transformers==4.51.3`, `deepspeed==0.16.4`, `peft==0.12.0`, `flash-attn==2.7.3`, `qwen-vl-utils==0.0.11`

---

## Weight Preparation

Both variants require DINOv2 ViT-L weights extracted from Depth-Anything-V2:

```bash
# Download depth_anything_v2_vitl.pth first, then:
python utils/process_vitl_weight.py \
  --dav2-vitl depth_anything_v2_vitl.pth \
  --dinov2-vitl weight/dino_vitl.pth
```

This strips the `pretrained.` prefix and removes the depth head, leaving only the ViT-L backbone weights.

Pre-trained model checkpoints on HuggingFace:
- `H-EmbodVis/Nautilus-llava-instruct-7b`
- `H-EmbodVis/Nautilus-qwen-instruct-7b`

---

## Training

### NAUTILUS-LLaVA
```bash
cd LLaVA
bash scripts/nautilus_finetune/finetune_nautilus_lora.sh
```
Config highlights: 4 GPUs, batch=128, 1 epoch, LoRA r=128 alpha=256, `nautilus_lr=2e-6` (VFE modules), `lr=2e-5` (projector + main).

### NAUTILUS-Qwen
```bash
cd qwen-vl-finetune
bash scripts/nautilus_finetune/nautilus_lora_sft_7b_lora.sh
```
Config highlights: 4-node torchrun, batch=4, 0.2 epochs, LoRA r=128 alpha=256, `nautilus_lr=2e-7` (much lower for VFE), `lr=2e-5` main.

**Important training design decisions:**
- DINOv2 encoder is always **frozen** — it is excluded from LoRA and gradients
- VFE modules use a separate, lower learning rate (`nautilus_lr`)
- LoRA targets all linear layers except vision encoder and Nautilus modules; Nautilus modules are in `modules_to_save` (fully trained)

---

## Inference

### NAUTILUS-LLaVA
```bash
cd LLaVA
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference.py \
  --model-path "path/to/checkpoint" \
  --dinov2-weight "weight/dino_vitl.pth" \
  --image "path/to/image.jpg" \
  --prompt "Describe the image"
```

### NAUTILUS-Qwen
```bash
cd qwen-vl-finetune
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint "path/to/checkpoint" \
  --image "path/to/image.jpg" \
  --prompt "Describe the image"
```

The Qwen inference script handles bounding box coordinate rescaling (original image space → model input space) and applies `double_image_tokens()` to account for the 2x token output of the VFE module.

---

## Evaluation

```bash
bash eval/eval_nautilus_llava.sh   # NAUTILUS-LLaVA
bash eval/eval_nautilus_qwen.sh    # NAUTILUS-Qwen
```

Evaluation scripts run inference on the NautData test split, route samples to task buckets via `sortbyid()` (using character at index 1 of the sample ID), then call per-task metric functions. Results can also be computed from a saved prediction JSON without re-running inference.

**Task ID mapping** (from `eval/utils.py`):
```
"0" → Image Caption
"1" → Grounding
"2" → Region Caption
"3" → VQA
"4" → Fishnet Classification
"5" → Detection
"6" → Counting
"7" → Region Classification
```

**Note:** Text metrics (BLEU, CIDEr, METEOR) require Java to be installed for the pycocoevalcap library.

---

## Data Format

Dataset annotations follow the LLaVA/Qwen conversation format:

```json
{
  "id": "X4...",
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nYour prompt here"},
    {"from": "gpt", "value": "Model response"}
  ]
}
```

The character at index 1 of `id` encodes the task type (see task ID mapping above). The dataset is `Nautilus_Instruct` (1.45M image-text pairs), available on the `dataset` branch. Percentage-based sampling is supported: `dataset_name%50` uses 50% of the data.

---

## No Test Suite

There is no `pytest`/`unittest` setup. Correctness validation is done entirely through the eval scripts. Use `qwen-vl-finetune/tools/check_image.py` to verify dataset image paths before training.

---

## Common Pitfalls

1. **Wrong environment**: LLaVA and Qwen branches have incompatible dependency versions (different torch, transformers, deepspeed). Always activate the correct conda env.

2. **Missing DINOv2 weights**: Both variants need `dino_vitl.pth`. Run `utils/process_vitl_weight.py` first.

3. **Qwen 2x token count**: The Qwen VFE module doubles the image token sequence. Any inference or evaluation code for Qwen must call `double_image_tokens()` from `eval/utils.py` to update `input_ids` accordingly.

4. **Bounding box coordinates**: Detection/grounding outputs are in model input resolution space. Use `scale_bboxes_in_text()` from `eval/utils.py` to rescale back to original image coordinates before evaluation.

5. **DINOv2 is frozen**: Do not add `nautilus_encoder` to trainable parameters. It should always be frozen.

6. **LoRA merging before deployment**: After training, merge LoRA weights into the base model using `LLaVA/scripts/merge_lora_weights.py` or `qwen-vl-finetune/scripts/merge_lora.py` before running standalone inference.
