"""Public single-question inference API."""

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.initialization import no_init_weights
from .checkpoint import checkpoint_file, read_metadata, load_weights
from .model.joint import LingBotNaviJointAdapter
from .prompts import make_messages, encode_prompt, canonical_goal
from .generation import generate_joint


def _load_memory(path, *, feature_dim, checkpoint_metadata):
    """Read final tensors only; raw blocks must go through build_memory first."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "memory.npz"
    if path.suffix != ".npz":
        raise ValueError("Inference requires a prepared memory.npz from build_memory")
    with np.load(path, allow_pickle=False) as data:
        required = {"memory_xyz", "memory_features", "block_cu_seqlens"}
        if not required.issubset(data.files):
            raise ValueError("Raw memory is not model input; build memory.npz first")
        xyz = np.asarray(data["memory_xyz"], dtype=np.float32)
        features = np.asarray(data["memory_features"], dtype=np.float32)
        offsets = np.asarray(data["block_cu_seqlens"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("Invalid prepared memory XYZ shape")
        if features.shape != (len(xyz), feature_dim):
            raise ValueError("Invalid prepared memory feature shape")
        if not np.isfinite(xyz).all() or not np.isfinite(features).all():
            raise ValueError("Nonfinite prepared memory")
        if offsets.dtype.kind not in "iu":
            raise ValueError("Block offsets must be integers")
        if (
            offsets.ndim != 1 or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(xyz)
            or np.any(offsets[1:] <= offsets[:-1])
            or len(xyz) > np.iinfo(np.int32).max
        ):
            raise ValueError("Invalid block offsets")
        # Legacy final NPZs have these tensors but no preparation metadata.
        if "metadata_json" in data:
            try:
                metadata = json.loads(data["metadata_json"].item())
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid prepared memory metadata") from exc
            if not isinstance(metadata, dict):
                raise ValueError("Invalid prepared memory metadata")
            block_ids = metadata.get("block_ids")
            if (
                metadata.get("format") != "lingbot-navi-memory-v1"
                or metadata.get("coordinate_frame") != "independent_block_local_meters"
                or metadata.get("feature_dim") != feature_dim
                or not isinstance(block_ids, list)
                or len(block_ids) != len(offsets) - 1
                or any(not isinstance(name, str) or not name for name in block_ids)
                or len(set(block_ids)) != len(block_ids)
            ):
                raise ValueError("Unsupported or inconsistent prepared memory metadata")
            augmentation = checkpoint_metadata["joint_protocol"].get("block_augmentation", {})
            expected = dict(
                voxel_size=checkpoint_metadata["voxel_size"],
                rotation_degrees=list(
                    augmentation.get("rotation_degrees", (0, 0, 180))
                    if augmentation.get("augment", True) else (0, 0, 0)
                ),
                seed=augmentation.get("seed", 42),
            )
            if metadata.get("preparation") != expected:
                raise ValueError(
                    "Memory preparation differs from checkpoint; rebuild with "
                    f"PreparationSettings(**{expected!r})"
                )
    return dict(
        memory_xyz=torch.from_numpy(xyz),
        memory_features=torch.from_numpy(features),
        block_cu_seqlens=torch.from_numpy(offsets.astype(np.int32)),
        scene_block_cu_seqlens=torch.tensor([0, len(offsets) - 1], dtype=torch.int32),
        query_scene_indices=torch.tensor([0], dtype=torch.long),
    )


class LingBotNavi:
    def __init__(self, model, processor, metadata):
        self.model = model
        self.processor = processor
        self.metadata = metadata

    @classmethod
    def from_pretrained(
        cls,
        weights_path,
        *,
        device="cuda:0",
    ):
        """Load a complete safetensors file and its adjacent processor files."""
        metadata = read_metadata(weights_path)
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass device='cpu' to load on CPU")
        processor_dir = checkpoint_file(weights_path).parent
        processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        ids = {
            token: processor.tokenizer.convert_tokens_to_ids(token)
            for token in metadata["joint_protocol"]["token_ids"]
        }
        if ids != metadata["joint_protocol"]["token_ids"]:
            raise ValueError("Processor / spatial token IDs differ from checkpoint")
        config = Qwen3_5Config.from_dict(metadata["base_config"])
        with no_init_weights():
            model = Qwen3_5ForConditionalGeneration._from_config(
                config,
                dtype=getattr(torch, metadata["base_dtype"]),
                attn_implementation="sdpa",
            )
        attach_adapter(model, processor, metadata)
        load_weights(model, weights_path, metadata)
        configure_runtime(model)
        model.to(device=device).eval().requires_grad_(False)
        return cls(model, processor, metadata)

    @torch.inference_mode()
    def predict(
        self,
        spatial_memory_path,
        question,
        *,
        image=None,
        task="qa",
        max_new_tokens=256,
        goal_xyz=None,
        current_pose=None,
        localization_answer=None,
        return_geometry=False,
        return_block_diagnostics=False,
        continuation_messages=None,
        candidate_centers=None,
    ):
        """Return one JSON-compatible answer. All XYZ outputs are first-block meters.

        For a grounding differences follow-up, keep the original question and
        memory. Pass the previous assistant reply and the user's follow-up in
        continuation_messages, and the first reply's predicted_centers in order.
        """
        if not isinstance(question, str):
            raise TypeError("Only one question string is supported")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if candidate_centers is not None and (task != "grounding" or not continuation_messages):
            raise ValueError("candidate_centers requires a grounding follow-up with conversation history")
        adapter = self.model.lingbotnavi_trajectory_adapter
        if return_geometry and adapter.active_point_head is None:
            raise ValueError("This checkpoint does not contain a point head")
        memory = _load_memory(
            spatial_memory_path,
            feature_dim=adapter.model_config.spatial_feature_dim,
            checkpoint_metadata=self.metadata,
        )
        messages = make_messages(
            question,
            image=image,
            task=task,
            goal_xyz=goal_xyz,
            current_pose=current_pose,
            localization_answer=localization_answer,
        )
        if continuation_messages is not None:
            if task not in ("grounding", "localization"):
                raise ValueError("Continuation history requires a retrieval task")
            messages.extend(continuation_messages)
        device = self.model.get_input_embeddings().weight.device
        dtype = self.model.get_input_embeddings().weight.dtype
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=False
        ):
            inputs = encode_prompt(
                self.model,
                self.processor,
                messages,
                image,
                max_context_tokens=adapter.max_context_tokens,
            )
            inputs.update({key: value.to(device) for key, value in memory.items()})
            result = generate_joint(
                adapter,
                **inputs,
                max_sampled_tokens=max_new_tokens,
                candidate_centers=[candidate_centers] if candidate_centers is not None else None,
                viewpoint_grounding=[task in ("qa", "grounding") and continuation_messages is None],
                goal_xyz=[canonical_goal(goal_xyz)] if goal_xyz is not None else None,
                initial_pose=[current_pose] if current_pose is not None else None,
                return_point_predictions=return_geometry,
                return_block_diagnostics=return_geometry or return_block_diagnostics,
            )[0]
        answer = asdict(result)
        answer["spatial_memory"] = {
            "block_count": len(memory["block_cu_seqlens"]) - 1,
            "point_count": len(memory["memory_xyz"]),
            "points_per_block": torch.diff(memory["block_cu_seqlens"]).tolist(),
        }
        return answer


def attach_adapter(model, processor, metadata):
    cfg = metadata["model_config"]
    text = model.model.language_model
    actual = (
        text.config.hidden_size,
        text.config.num_hidden_layers,
        text.config.num_attention_heads,
        text.config.num_key_value_heads,
        text.config.head_dim,
    )
    saved = (
        cfg.qwen_hidden_dim,
        cfg.qwen_num_layers,
        cfg.qwen_num_heads,
        cfg.qwen_num_key_value_heads,
        cfg.qwen_head_dim,
    )
    if actual != saved:
        raise ValueError(f"Base architecture mismatch: {actual} != {saved}")
    if cfg.enforce_supported_shape and actual != (2560, 32, 16, 4, 256):
        raise ValueError("This release supports Qwen3.5-4B")
    if model.get_input_embeddings().num_embeddings < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    model.requires_grad_(False)
    adapter = LingBotNaviJointAdapter(
        text,
        root_model=model,
        tokenizer=processor.tokenizer,
        model_config=cfg,
        protocol=metadata["joint_protocol"],
    )
    # Adapter projections/heads keep their original FP32 parameter precision.
    model.lingbotnavi_trajectory_adapter = adapter


def configure_runtime(model):
    text = model.model.language_model
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for layer in text.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        module = layer.linear_attn
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = qwen.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
        # Override automatic fused normalization when kernels are installed.
        norm = qwen.Qwen3_5RMSNormGated(
            module.head_v_dim, eps=module.layer_norm_epsilon
        )
        norm.weight = module.norm.weight
        module.norm = norm
