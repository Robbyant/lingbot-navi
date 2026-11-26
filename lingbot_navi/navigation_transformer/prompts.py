"""Single-question prompts and native Qwen3.5 multimodal preprocessing."""

import math
from pathlib import Path
import torch
from PIL import Image
from .feedback import FeedbackPolicy, format_feedback

FRAME_INSTRUCTION = "Coordinates are in the first memory block frame.\n"
ORDERED_OBJECT_INSTRUCTION = (
    "\nReturn objects in ascending center-coordinate order (x, then y, then z). "
    "Coordinate tuples are in meters."
)
LOCALIZATION_QUESTION = (
    "Where was this image taken in the available spatial memory? Return its pose."
)


ABSENT = "I could not find any object matching that description in the available scene observations."
IMAGE_ABSENT = "This image cannot be located in the available spatial memory."


def frontier_messages(current_xyz, *, image_target=False, absence=None):
    canonical_goal(current_xyz)  # Validate without changing the two-decimal frontier prompt.
    position = "(" + ", ".join(f"{v:.2f}" for v in current_xyz) + ")"
    phrase = "the location shown in the image" if image_target else "the described target"
    question = f"My current position is {position}. Which frontier should I explore next to find {phrase}? Return its position."
    return [
        {"role": "assistant", "content": absence if absence is not None else IMAGE_ABSENT if image_target else ABSENT},
        {"role": "user", "content": question},
    ]


def canonical_goal(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(type(x) in (int, float) and math.isfinite(x) for x in value)
    ):
        raise ValueError("goal_xyz must contain three finite coordinates")
    return [round(float(x), 6) or 0.0 for x in value]


def candidate_prefixes(centers):
    if not 1 <= len(centers) <= 26:
        raise ValueError("Expected 1..26 candidate centers")
    return [
        chr(65 + i)
        + ": "
        + format_feedback("center", torch.tensor(c), FeedbackPolicy()).removeprefix(
            " at "
        )
        + " "
        for i, c in enumerate(centers)
    ]


def make_messages(
    question,
    *,
    image=None,
    task="qa",
    goal_xyz=None,
    current_pose=None,
    localization_answer=None,
):
    if task not in ("qa", "grounding", "localization", "coordinate"):
        raise ValueError("task must be qa, grounding, localization, or coordinate")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be one nonempty string")
    if task == "grounding":
        question = (
            "Find all objects matching the following description in the available scene observations.\n\n"
            "<description>\n"
            + question.strip()
            + "\n</description>"
            + ORDERED_OBJECT_INSTRUCTION
        )
    if task in ("localization", "coordinate") and image is None:
        raise ValueError("This task requires a current image")
    if task == "localization":
        question = LOCALIZATION_QUESTION
    if task == "coordinate":
        goal = canonical_goal(goal_xyz)
        if current_pose is None or not localization_answer:
            raise ValueError(
                "Coordinate continuation requires current_pose and localization_answer from localization"
            )
        # Match the source coordinate-navigation prompt exactly.
        question = (
            "Plan a route to ("
            + ", ".join(f"{x:.6f}" for x in goal)
            + ") meters in the first memory block coordinate frame."
        )
        first = LOCALIZATION_QUESTION
    else:
        if (
            goal_xyz is not None
            or current_pose is not None
            or localization_answer is not None
        ):
            raise ValueError(
                "Pose/goal continuation fields are only valid for task=coordinate"
            )
        first = question
    first = (
        FRAME_INSTRUCTION
        + first.replace("<|scene|>", "").replace("<image>", "").lstrip()
    )
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": first})
    messages = [{"role": "user", "content": content}]
    if task == "coordinate":
        messages.extend(
            [
                {"role": "assistant", "content": localization_answer},
                {"role": "user", "content": question},
            ]
        )
    return messages


@torch.inference_mode()
def encode_prompt(model, processor, messages, image, *, max_context_tokens):
    image_data = None
    if image is not None:
        if isinstance(image, Image.Image):
            image_data = image.convert("RGB")
        else:
            with Image.open(Path(image).expanduser()) as img:
                image_data = img.convert("RGB")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        text=[text],
        images=[image_data] if image_data is not None else None,
        return_tensors="pt",
        padding=False,
    )
    device = model.get_input_embeddings().weight.device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    ids = inputs["input_ids"]
    if ids.shape[0] != 1 or ids.shape[1] >= max_context_tokens:
        raise ValueError("Expected one prompt shorter than max_context_tokens")
    embeds = model.get_input_embeddings()(ids)
    if image_data is not None:
        visual = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        )
        features = torch.cat(visual.pooler_output, dim=0).to(embeds)
        image_mask = ids.eq(model.config.image_token_id)
        if int(image_mask.sum()) != features.shape[0]:
            raise ValueError("Image token/feature count mismatch")
        embeds = embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeds), features
        )
    modalities = inputs.get("mm_token_type_ids")
    if modalities is None:
        modalities = ids.eq(model.config.image_token_id).long()
    positions, _ = model.model.get_rope_index(
        ids,
        mm_token_type_ids=modalities,
        image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=inputs["attention_mask"],
    )
    return dict(
        prompt_input_ids=ids,
        prompt_inputs_embeds=embeds,
        prompt_attention_mask=inputs["attention_mask"],
        prompt_position_ids=positions,
    )
