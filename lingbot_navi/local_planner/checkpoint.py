"""Load the released local head and verify its frozen image encoder."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .vision import VisualLocalPlanner, digest, file_digest, load_current_image_processor
from lingbot_navi.vision import load_vision

SUPPORTED_SCHEMA = "lingbotnavi_encoder_style_self_attention_head_v1"
SUPPORTED_HEAD = {
    "local_planner_head_type": "encoder_style_self_attention",
    "use_prompt_features": False,
    "navigation_contract": "current_ego_v1",
    "conditioning_mode": "route_history_v1",
    "history_token_order": "prompt_images_navigation_queries_v1",
    "route_point_encoding": "metric_fourier_xyz_arc_order_v1",
    "action_query_distance_encoding": "slot_order_v1",
    "meta_action_mode": "turn_arrival_v1",
    "meta_action_include_stop": False,
    "attention_mode": "prefix_bidirectional",
    "horizon": 10,
}
SUPPORTED_ROUTE = {
    "route_source": "qwen",
    "route_scope": "reference_suffix",
    "global_points": 16,
    "route_crop_ahead_m": 4,
    "route_crop_behind_m": 0.5,
    "history_policy": "pose_keyframes_v1",
}


def _require_fields(actual, expected):
    for name, value in expected.items():
        if name not in actual or actual[name] != value:
            raise ValueError(f"Unsupported local checkpoint {name}: expected {value!r}")


def load_model(checkpoint_path, vision_path, device, expected_sha256=None):
    checkpoint_path = Path(checkpoint_path).expanduser()
    vision_path = Path(vision_path).expanduser()
    if expected_sha256 is not None and file_digest(checkpoint_path) != expected_sha256:
        raise ValueError("Local checkpoint SHA-256 mismatch")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require_fields(saved, {"schema_version": SUPPORTED_SCHEMA, "head_format": "trainable_only"})
    contract = saved["contract"]
    runtime = contract["runtime"]
    config = runtime["model_config"]
    _require_fields(config, SUPPORTED_HEAD)
    route = config["route_config"]
    _require_fields(route, SUPPORTED_ROUTE)
    data = contract["data_config"]
    if (tuple(data["ego_in_recorded_position_m"]) != (0, 0, 0)
            or tuple(data["ego_in_recorded_rotation_xyzw"]) not in ((0, 0, 0, 1), (0, 0, 0, -1))):
        raise ValueError("Local checkpoint requires an unsupported camera/body transform")
    if runtime.get("qwen_source_identity") is not None:
        raise ValueError("Local head must use the independent encoder architecture")

    identity = runtime["prompt_encoder_identity"]
    config_path = vision_path / "config.json"
    base = json.loads(config_path.read_text())
    if (base["vision_config"]["out_hidden_size"] != config["vision_width"]
            or base["text_config"]["hidden_size"] != config["prompt_width"]):
        raise ValueError("Qwen source feature widths do not match local checkpoint")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[runtime["vision_dtype"]]
    vision, vision_identity = load_vision(vision_path, device, dtype)
    # The training checkpoint fingerprints the compact Qwen vision config. HF
    # exports also serialize inherited defaults and the resolved vision type.
    if vision_identity["config_sha256"] != identity["visual_identity"]["config_sha256"]:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

        compact = dict(base["vision_config"])
        defaults = Qwen3_5VisionConfig().to_dict()
        for key in ("return_dict", "chunk_size_feed_forward", "label2id", "output_attentions",
                    "output_hidden_states", "is_encoder_decoder", "architectures", "problem_type",
                    "id2label"):
            if key in compact and digest(compact[key]) == digest(defaults.get(key)):
                del compact[key]
        if compact.get("dtype") == runtime["vision_dtype"]:
            del compact["dtype"]
        if compact.get("model_type") == Qwen3_5VisionConfig.model_type:
            compact["model_type"] = "qwen3_5"
        vision_identity["config_sha256"] = digest(compact)
    if vision_identity != runtime["vision_identity"] or vision_identity != identity["visual_identity"]:
        raise ValueError("Qwen visual weights do not match local checkpoint")
    processor = load_current_image_processor(vision_path, identity, max_pixels=runtime["max_pixels"])
    if digest(json.loads(processor.to_json_string())) != runtime["processor_sha256"]:
        raise ValueError("Image processor does not match local checkpoint")

    model = VisualLocalPlanner(vision, SimpleNamespace(**config)).to(device)
    state = saved["head"]
    expected = dict(model.head.named_parameters())
    if set(expected) != set(state):
        raise ValueError(f"Local head parameters differ: missing={sorted(set(expected) - set(state))}, "
                         f"unexpected={sorted(set(state) - set(expected))}")
    for name, value in state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Incompatible local head parameter: {name}")
    model.head.load_state_dict(state, strict=False)  # Query progress is a deterministic buffer.
    backend = runtime.get("planning_head_attention", {}).get("backend", "auto")
    if backend not in ("auto", "math"):
        raise ValueError(f"Unsupported planning head attention backend: {backend}")
    return model.eval().requires_grad_(False), processor, SimpleNamespace(**route), backend
