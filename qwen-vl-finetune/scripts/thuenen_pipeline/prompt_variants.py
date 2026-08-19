"""The prompt / few-shot variant registry for the Thünen screening sweep.

One builder per variant, each returning a ``messages`` list for
``processor.apply_chat_template``. Deliberately **torch-free and import-light** so
the whole registry can be read, diffed and rendered on a host without torch --
these are the experiment's independent variable, and they need to be reviewable
without a GPU box in the loop.

Two families:

*Prompt-only* (``P*``) send one image, the query, and vary only the instruction.
Every one of them is a single edit away from ``P0_baseline``, which is
``nautilus_zeroshot.PROMPT_TEMPLATE`` imported verbatim, so any movement is
attributable to that edit.

*Few-shot* (``F*``) prepend demonstration turns before the query. ``F1``/``F2``/``F3``
use genuine ``user(image + instruction) -> assistant(bbox JSON)`` pairs, which is what
the earlier attempt (``infer_fewshot.build_messages``) never did -- it packed intro,
exemplars, captions and query into a single user turn, so the model was never shown
one example of the input->output mapping it was being asked to imitate. The assistant
turns reproduce the model's own output format exactly (one JSON object per line,
comma-terminated except the last, no wrapping array), because a demonstration in a
format the model does not emit teaches the format, not the task.

Instruction held constant
-------------------------
Every ``F*`` variant repeats the ``P0`` instruction on every turn, so ``F*`` vs ``P0``
isolates the demonstrations and nothing else.

Note on ``P4``
--------------
The plan called this ``P4_no_format_hint``, on the assumption that the baseline
carries a JSON schema example to drop. It does not -- ``PROMPT_TEMPLATE`` is a bare
sentence with no schema. The informative test of format sensitivity is therefore the
other direction: **add** the explicit schema that ``infer_fewshot``'s default prompt
uses. Hence ``P4_json_schema``.

Coordinates
-----------
Builders render whatever numbers they are handed. The runner passes exemplars whose
``bbox_2d`` values have already been rescaled into *that exemplar's own* model-input
pixel space -- each image in a multi-image message is resized independently, and the
model reads and writes coordinates in model-input space, not original pixels.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nautilus_zeroshot import PROMPT_TEMPLATE  # noqa: E402

# Short visual phrases, one per prompt class, in classes_prompt.txt order. Used by
# P3 (inlined in place of the names) and F4 (as a field-guide preamble). Written to
# separate the confusable pairs the zero-shot run collapsed -- brittle star vs
# starfish vs sand star, masked crab vs swimming crab vs hermit crab.
DESCRIPTIONS = {
    "brittle star": "a small round central disc with five long, thin, snake-like arms much narrower than the disc",
    "starfish": "a thick five-armed star whose arms are broad where they join the body, the whole animal one solid piece",
    "tower shell snail": "a long, narrow, pointed spiral shell shaped like a screw or steeple",
    "bivalve shell": "a pair of hinged oval shells, often half-buried in the sediment",
    "swimming crab": "a crab whose hind legs end in flattened paddles",
    "comb jelly": "a transparent gelatinous oval drifting in the water above the seabed",
    "sea anemone": "a soft column crowned with a ring of tentacles, fixed to the bottom",
    "unidentified organism": "an animal that is clearly present but fits none of the other names",
    "masked crab": "a pale oval crab with two very long antennae, usually buried with only the antennae showing",
    "sand eel": "a very slender, elongated, silvery fish, often part-buried in sand",
    "sand star": "a five-armed star with sharply pointed arms edged with regular comb-like spines",
    "flatfish": "an oval, flattened fish lying flat against the seabed, both eyes on the upper side",
    "dragonet": "a small bottom-sitting fish with a flattened head and large fanned-out pectoral fins",
    "hermit crab": "a crab carrying and living inside an empty snail shell",
    "fish": "a fish swimming freely in the water above the seabed",
    "animal track": "a furrow, trail or burrow opening left in the sediment -- a mark, not an animal",
    "pipefish": "an extremely thin, stiff, stick-like fish with a long tubular snout",
    "goby": "a small, stout, blunt-headed fish resting directly on the bottom",
    "echinoderm": "a spiny-skinned animal that cannot be placed more precisely",
    "shrimp": "a small translucent long-bodied crustacean with long thin antennae",
    "razor clam": "a long, straight, narrow shell shaped like a cut-throat razor",
    "trough shell clam": "a smooth, thick, roughly triangular clam shell",
    "sea urchin": "a round spiny ball sitting on the bottom",
    "hooknose fish": "a small armoured bottom-living fish covered in bony plates",
}


class Context(object):
    """Everything a builder may need, assembled once by the runner."""

    def __init__(self, prompt_names, query_max_pixels, exemplar_max_pixels,
                 exemplars=None, annotate=None):
        self.prompt_names = prompt_names
        self.query_max_pixels = query_max_pixels
        self.exemplar_max_pixels = exemplar_max_pixels
        self.exemplars = exemplars or []
        # Callable(exemplar) -> path of a copy with its boxes burnt in as red
        # rectangles. Only F0 uses it; the runner supplies and caches it.
        self.annotate = annotate


# ── shared pieces ────────────────────────────────────────────────────────────
def class_list(ctx):
    return ", ".join(ctx.prompt_names)


def baseline_prompt(ctx):
    """``nautilus_zeroshot.PROMPT_TEMPLATE``, the reference instruction."""
    return PROMPT_TEMPLATE.format(classes=class_list(ctx))


def query_turn(ctx, query_path, text):
    """The query image carries **no** ``max_pixels``, deliberately.

    ``batch_inference.run_inference`` -- the path every existing NAUTILUS number in
    this thesis was produced by -- omits it and lets the processor's global
    ``max_pixels`` do the final resize. That resamples twice (2704x1520 -> 2716x1512
    by ``process_vision_info``'s default, then -> 1344x756 by the processor), where
    passing ``max_pixels`` here would resample once, straight to 1344x756.

    Both land on the *same* 54x96 grid, so nothing downstream complains -- but the
    pixels differ, the logits differ, and greedy decoding then diverges completely.
    Measured: 0 of 12 images produced identical output across the two paths, against
    a same-GPU noise floor of 12 of 12 identical. Matching the house path is what
    makes P0_baseline reproduce the finished full-split run, and what keeps every
    variant's query treated exactly as the baseline's.

    Exemplars *do* pass ``max_pixels``: they need a smaller budget than the query and
    have no prior convention to match.
    """
    return {"role": "user", "content": [
        {"type": "image", "image": str(query_path)},
        {"type": "text", "text": text},
    ]}


def single(ctx, query_path, text):
    """The prompt-only message shape: one image, one instruction."""
    return [query_turn(ctx, query_path, text)]


def render_boxes(boxes):
    """The model's own output format, reproduced exactly.

    One JSON object per line, comma-terminated except the last, with no wrapping
    array -- copied from what the checkpoint actually emits (see any file under
    ``runs/thuenen_zeroshot/results``). Demonstrations in a shape the model does not
    produce would teach the shape instead of the task.
    """
    lines = ['{{"bbox_2d": [{}, {}, {}, {}], "label": "{}"}}'.format(
        *(list(box["bbox_2d"]) + [box["label"]])) for box in boxes]
    return ",\n".join(lines)


# ── prompt-only variants ─────────────────────────────────────────────────────
def build_p0(ctx, query_path):
    return single(ctx, query_path, baseline_prompt(ctx))


def build_p1(ctx, query_path):
    numbered = "\n".join("{:2d}. {}".format(i, name)
                         for i, name in enumerate(ctx.prompt_names, start=1))
    text = (
        "Detect all objects in the image and output their locations as bounding "
        "boxes with labels.\n\n"
        "The label of every box must be copied verbatim from this list of {n} "
        "names, and from nowhere else:\n{numbered}\n\n"
        "Do not invent a label. Do not use a synonym, a common name or a shorter "
        "word that is not on the list. If an animal is present but matches none of "
        "the {n} names, label it \"unidentified organism\"."
    ).format(n=len(ctx.prompt_names), numbered=numbered)
    return single(ctx, query_path, text)


def build_p2(ctx, query_path):
    text = (
        "Detect every animal in the image and output the location of each as a "
        "bounding box. Do not identify the species. Give every single box the same "
        "label: \"organism\"."
    )
    return single(ctx, query_path, text)


def build_p3(ctx, query_path):
    described = "; ".join('{} ({})'.format(name, DESCRIPTIONS[name])
                          for name in ctx.prompt_names)
    text = (
        "Detect all objects in the image and output their locations as bounding "
        "boxes with labels. Possible objects, each followed by what it looks like, "
        "are: {described}. Use the name, not the description, as the label."
    ).format(described=described)
    return single(ctx, query_path, text)


def build_p4(ctx, query_path):
    text = (
        "Detect all objects in the image and output their locations as bounding "
        "boxes in JSON format, one object per line, e.g. "
        "{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"...\"}}. "
        "Possible Objects are {classes}"
    ).format(classes=class_list(ctx))
    return single(ctx, query_path, text)


def build_p5(ctx, query_path):
    text = (baseline_prompt(ctx) + ". If none of these objects is present in the "
            "image, output exactly: []")
    return single(ctx, query_path, text)


def build_p7(ctx, query_path):
    text = (baseline_prompt(ctx) + ". Most images contain between 0 and 3 objects. "
            "Output a box only where you are confident an object is really there; "
            "it is better to return nothing than to guess.")
    return single(ctx, query_path, text)


# ── few-shot variants ────────────────────────────────────────────────────────
def build_f0(ctx, query_path):
    """The old single-user-turn, red-rectangle, captioned format, as a control.

    Reproduces ``infer_fewshot.build_messages`` on the screening subsample so the
    one-off two-image result has a matched row in the same table. Everything sits in
    one user turn and there is no demonstrated output -- that is the point.
    """
    content = [{"type": "text", "text":
                "Here are example images. In each one, the target species is marked "
                "with a red rectangle."}]
    for i, exemplar in enumerate(ctx.exemplars, start=1):
        content.append({"type": "image", "image": ctx.annotate(exemplar),
                        "max_pixels": ctx.exemplar_max_pixels})
        labels = ", ".join(sorted({box["label"] for box in exemplar["boxes"]}))
        content.append({"type": "text",
                        "text": "Example {}: the species in the red rectangles are "
                                "{}.".format(i, labels)})
    content.append({"type": "text", "text": "Now consider this new image:"})
    # No max_pixels -- see query_turn.
    content.append({"type": "image", "image": str(query_path)})
    content.append({"type": "text", "text": baseline_prompt(ctx)})
    return [{"role": "user", "content": content}]


def make_icl(k):
    """Build a k-shot multi-turn in-context-learning variant.

    Alternating ``user(image + the P0 instruction) -> assistant(bbox JSON)`` pairs,
    then the query asked with the identical instruction. Holding the instruction
    fixed across all k+1 turns is what makes ``F* - P0`` the effect of the
    demonstrations alone.
    """

    def build(ctx, query_path):
        instruction = baseline_prompt(ctx)
        messages = []
        for exemplar in ctx.exemplars[:k]:
            messages.append({"role": "user", "content": [
                {"type": "image", "image": exemplar["image"],
                 "max_pixels": ctx.exemplar_max_pixels},
                {"type": "text", "text": instruction},
            ]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": render_boxes(exemplar["boxes"])},
            ]})
        messages.append(query_turn(ctx, query_path, instruction))
        return messages

    return build


def build_f4(ctx, query_path):
    """No exemplar images -- a text field guide, then the baseline instruction.

    Splits what ``F1``/``F2``/``F3`` confound: whether any conditioning gets through
    at all, or only conditioning that arrives as pixels.
    """
    guide = "\n".join("- {}: {}".format(name, DESCRIPTIONS[name])
                      for name in ctx.prompt_names)
    text = ("Field guide to the species in these images:\n{guide}\n\n{instruction}"
            .format(guide=guide, instruction=baseline_prompt(ctx)))
    return single(ctx, query_path, text)


# ── registry ─────────────────────────────────────────────────────────────────
# ``exemplars`` is how many exemplar frames the variant needs, which tells the
# runner which exemplars_k<K>.json to load. ``annotated`` marks the one variant that
# needs boxes burnt into the exemplar images instead of written as coordinates.
VARIANTS = {
    "P0_baseline":     {"exemplars": 0, "build": build_p0,
                        "note": "nautilus_zeroshot.PROMPT_TEMPLATE verbatim"},
    "P1_closed_vocab": {"exemplars": 0, "build": build_p1,
                        "note": "numbered list + hard no-invention constraint"},
    # snap=False: this variant's labels are deliberately outside the prompt
    # vocabulary, and snap_labels does substring matching -- "organism" is a
    # substring of "unidentified organism", so snapping would silently rewrite every
    # box into a class the instruction never asked for. P2 is a localization
    # measurement; its class-aware score is meaningless by construction.
    "P2_agnostic":     {"exemplars": 0, "build": build_p2, "snap": False,
                        "note": "localization ceiling, naming removed"},
    "P3_descriptive":  {"exemplars": 0, "build": build_p3,
                        "note": "visual phrases inline, canonical name as label"},
    "P4_json_schema":  {"exemplars": 0, "build": build_p4,
                        "note": "explicit output schema (see module docstring)"},
    "P5_abstain":      {"exemplars": 0, "build": build_p5,
                        "note": "baseline + explicit empty-output option"},
    "P7_sparse_prior": {"exemplars": 0, "build": build_p7,
                        "note": "baseline + sparse-scene prior"},
    "F0_caption_control": {"exemplars": 2, "build": build_f0, "annotated": True,
                           "note": "old one-turn captioned format, matched control"},
    "F1_icl_k2":       {"exemplars": 2, "build": make_icl(2),
                        "note": "2 demonstration turns"},
    "F2_icl_k4":       {"exemplars": 4, "build": make_icl(4),
                        "note": "4 demonstration turns"},
    "F3_icl_k8":       {"exemplars": 8, "build": make_icl(8),
                        "note": "8 demonstration turns"},
    "F4_field_guide":  {"exemplars": 0, "build": build_f4,
                        "note": "text descriptions, no exemplar images"},
}

# The order the report table is written in, and the default sweep.
SWEEP = ["P0_baseline", "P1_closed_vocab", "P2_agnostic", "P3_descriptive",
         "P4_json_schema", "P5_abstain", "P7_sparse_prior",
         "F0_caption_control", "F1_icl_k2", "F2_icl_k4", "F3_icl_k8",
         "F4_field_guide"]


def scale_exemplars(exemplars, scales):
    """Copy ``exemplars`` with each one's boxes moved into its own model-input space.

    ``scales`` is one ``(w_scale, h_scale)`` per exemplar, read from that exemplar's
    own ``image_grid_thw`` row. Scaling the numbers directly rather than regexing the
    rendered text (``infer_fewshot.scale_bboxes_in_text``) avoids a round-trip through
    string formatting; the int rounding is the same either way, and the model's
    coordinate vocabulary is integers.
    """
    out = []
    for exemplar, (w_scale, h_scale) in zip(exemplars, scales):
        boxes = []
        for box in exemplar["boxes"]:
            x1, y1, x2, y2 = box["bbox_2d"]
            boxes.append({"bbox_2d": [int(round(x1 * w_scale)), int(round(y1 * h_scale)),
                                      int(round(x2 * w_scale)), int(round(y2 * h_scale))],
                          "label": box["label"]})
        copy = dict(exemplar)
        copy["boxes"] = boxes
        out.append(copy)
    return out
