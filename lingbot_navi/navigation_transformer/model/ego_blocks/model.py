import torch
from torch import nn
from ..configuration import ModelConfig, BlockConfig
from ..coordinate_encoding import (
    SpatialPositionEncoder,
    ordered_query_progress_encoding,
)
from ....spatial_encoding import SpatialRoPESpec
from .backbone import EgoBlockQwen35
from ..lora import LoRALinear
from .pi3_pose import Pi3PoseHead
from .full_spatial import Float32Module


class EgoBlockEmbedding(Float32Module):
    def __init__(self, model_config, task_config):
        super().__init__()
        width = model_config.qwen_hidden_dim
        self.block_token = nn.Parameter(torch.randn(1, width) * 0.02)
        self.anchor_block_token = nn.Parameter(torch.randn(1, width) * 0.02)
        self.anchor_role = nn.Parameter(torch.randn(1, width) * 0.02)
        self.goal_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.goal_role = nn.Parameter(torch.randn(1, width) * 0.02)
        self.trajectory_queries = nn.Parameter(
            torch.randn(task_config.num_interior_points, width) * 0.02
        )
        self.position_gate = nn.Parameter(
            torch.tensor([task_config.position_gate_init])
        )
        spec = SpatialRoPESpec(
            model_config.spatial_rope_min_one_rad_distance_m,
            model_config.spatial_rope_max_one_rad_distance_m,
        )
        self.position_encoder = SpatialPositionEncoder(
            width, task_config.hidden_dim, coordinate_spec=spec
        )
        self.register_buffer(
            "progress",
            ordered_query_progress_encoding(task_config.num_interior_points, width),
            persistent=False,
        )

    def block(self, features, xyz, *, anchor=False):
        points = features.to(
            self.block_token
        ) + self.position_gate * self.position_encoder(xyz).to(self.block_token)
        tokens = torch.cat(
            (self.anchor_block_token if anchor else self.block_token, points)
        )
        return tokens + self.anchor_role if anchor else tokens

    @property
    def trajectory_progress_encoding(self):
        return self.progress


class LingBotNaviEgoBlockModel(nn.Module):
    def __init__(
        self,
        text_model: nn.Module,
        *,
        model_config=None,
        task_config=None,
    ):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.task_config = task_config or BlockConfig()
        self.text_model = text_model
        if (
            int(text_model.config.hidden_size) != self.model_config.qwen_hidden_dim
            or len(text_model.layers) != self.model_config.qwen_num_layers
        ):
            raise ValueError("text backbone shape differs from model_config")
        self.text_model.requires_grad_(False)
        self.backbone = EgoBlockQwen35(
            text_model,
            self.model_config,
            spatial_full=self.task_config.spatial_full,
        )
        routes = ("prompt", "trajectory") if self.task_config.spatial_full else (
            "prompt", "spatial", "trajectory"
        )
        for index in self.model_config.full_attention_layers:
            attention = text_model.layers[index].self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attention, name, LoRALinear(getattr(attention, name), routes, self.model_config))
        self.prompt_embedding = EgoBlockEmbedding(self.model_config, self.task_config)
        width, hidden = (self.model_config.qwen_hidden_dim, self.task_config.hidden_dim)
        self.goal_head = Pi3PoseHead(width)
        self.block_pose_head = Pi3PoseHead(
            width, predict_scale=self.task_config.learn_block_scale
        )
        self.trajectory_norm = nn.LayerNorm(width)
        self.trajectory_head = nn.Sequential(
            nn.Linear(width + 4, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.register_buffer(
            "trajectory_progress",
            torch.linspace(0.0, 1.0, 80)[1:-1, None],
            persistent=False,
        )
        reference = next(text_model.parameters())
        self.prompt_embedding.to(device=reference.device)
        for module in (
            self.goal_head,
            self.block_pose_head,
            self.trajectory_norm,
            self.trajectory_head,
        ):
            module.to(device=reference.device)
        self._keep_heads_fp32()

    def _keep_heads_fp32(self):
        for module in (
            self.goal_head,
            self.block_pose_head,
            self.trajectory_norm,
            self.trajectory_head,
        ):
            module.float()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._keep_heads_fp32()
        return self

    def _initial_memory(self, xyz, features, point_bounds, block_start, block_end):
        embeddings, positions, bounds = ([], [], [0])
        for block in range(block_start, block_end):
            start, end = point_bounds[block : block + 2]
            embeddings.append(
                self.prompt_embedding.block(
                    features[start:end], xyz[start:end], anchor=block == block_start
                )
            )
            positions.append(torch.cat((xyz.new_zeros(1, 3), xyz[start:end])))
            bounds.append(bounds[-1] + end - start + 1)
        return (
            torch.cat(embeddings)[None],
            torch.cat(positions),
            torch.tensor(bounds, dtype=torch.int32, device=xyz.device),
        )
