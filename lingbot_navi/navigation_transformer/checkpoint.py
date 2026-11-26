"""Load complete LingBot-Navi weights without a separate base-model checkpoint."""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from .constants import SPATIAL_SPECIAL_TOKENS, VIEWPOINT_SPECIAL_TOKENS, TRAJECTORY_FLOW_PROTOCOL
from .model.configuration import ModelConfig

CHECKPOINT_FORMAT = "lingbotnavi_full_v1"
JOINT_PROTOCOL = "lingbotnavi_first_block_joint_v3"
VIEWPOINT_PROTOCOL = "lingbotnavi_first_block_joint_v4"


def checkpoint_file(path):
    path = Path(path).expanduser().resolve()
    return path / "model.safetensors" if path.is_dir() else path


def validate_metadata(raw):
    protocol = raw["joint_protocol"]
    version = protocol.get("version")
    if version not in (JOINT_PROTOCOL, VIEWPOINT_PROTOCOL):
        raise ValueError(f"Unsupported joint protocol: {version}")
    if protocol.get("trajectory_flow") not in (None, TRAJECTORY_FLOW_PROTOCOL):
        raise ValueError("Unsupported trajectory flow protocol")
    viewpoint = version == VIEWPOINT_PROTOCOL
    expected = {
        "version": version,
        "coordinate_frame": "first_block_meters",
        "memory_conditioning": "memory_only_shared_prefix",
        "trajectory_attention": "bidirectional_full_attention_queries_only",
        "trajectory_lora": "independent_qkvo",
        "trajectory_gla": "causal_shared_frozen",
        "trajectory_query_count": 78,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Incompatible model protocol {key}: {protocol.get(key)!r}")
    if protocol.get("lm_head_mode") not in ("split", "unified"):
        raise ValueError("Unsupported LM readout")
    if set(protocol["token_ids"]) != set(VIEWPOINT_SPECIAL_TOKENS if viewpoint else SPATIAL_SPECIAL_TOKENS):
        raise ValueError("Spatial token vocabulary mismatch")
    if viewpoint:
        expected_viewpoint = dict(version="occ_first_exit_source_yaw_v1", offset_m=0.25,
            layout="xyz_cos_sin", heading_frame="project_world_xy_to_first_block_xy",
            feedback=False, missing="numeric_loss_mask")
        if protocol.get("viewpoint") != expected_viewpoint:
            raise ValueError("Unsupported viewpoint protocol")
    if protocol.get("block_task_config", {}).get("learn_block_scale", False):
        raise ValueError("Only metric, unit-scale spatial memory is supported")
    if protocol.get("block_augmentation", {}).get("scale_augment", False):
        raise ValueError("Scale augmentation is incompatible with metric inference")
    return {**raw, "model_config": ModelConfig.from_dict(raw["model_config"])}


def read_metadata(path):
    with safe_open(str(checkpoint_file(path)), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    key = "lingbotnavi" if "lingbotnavi" in header else "ling_nav"
    if key not in header:
        raise ValueError("Expected a complete LingBot-Navi checkpoint with embedded metadata")
    raw = json.loads(header[key])
    if key == "ling_nav" and raw.get("format") == "ling_nav_full_v1":
        raw["format"] = CHECKPOINT_FORMAT
        version = raw["joint_protocol"]["version"]
        if version.startswith("ling_nav_"):
            raw["joint_protocol"]["version"] = "lingbotnavi_" + version[len("ling_nav_"):]
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected unmerged lingbotnavi_full_v1 weights; re-export the original checkpoint")
    if raw.get("projection_layout", "route_specific_lora") != "route_specific_lora":
        raise ValueError("Only unmerged LoRA projections are supported")
    if raw.get("base_dtype") not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported base-model precision")
    return validate_metadata(raw)


def canonical_parameter_name(name):
    namespace, separator, parameter = name.partition(".")
    if separator and namespace.endswith("_trajectory_adapter"):
        return "lingbotnavi_trajectory_adapter." + parameter
    return name


def normalize_parameter_names(state):
    renamed = {canonical_parameter_name(k): v for k, v in state.items()}
    if len(renamed) != len(state):
        raise ValueError("Parameter rename collision")
    return renamed


def load_weights(model, path, metadata):
    state = load_file(str(checkpoint_file(path)), device="cpu")
    aliases = metadata.get("shared_tensors", {})
    for name, target in aliases.items():
        if name in state or target not in state or target in aliases:
            raise ValueError(f"Invalid shared tensor: {name} -> {target}")
    state.update({name: state[target] for name, target in aliases.items()})
    state = normalize_parameter_names(state)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"Weight names differ: missing={sorted(set(expected) - set(state))}, "
            f"extra={sorted(set(state) - set(expected))}"
        )
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Weight shape mismatch: {name}")
    # Preserve each saved tensor's precision, including FP32 task parameters.
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return len(state)
