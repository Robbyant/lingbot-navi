import copy
import itertools
import torch
from torch import Tensor, nn
from ..attention import dense_attention, varlen_attention
from ..lora import LoRALinear
from typing import Literal
from ....spatial_encoding import SpatialRoPESpec
from ..spatial_rope import spatial_xyz_position_embeddings
from .inputs import image_span, offsets
from .full_spatial import FullSpatialModule, SpatialLinear


ProjectionRoute = Literal["prompt", "spatial", "trajectory"]


def _apply_rotary_value(
    value: Tensor, position_embeddings: tuple[Tensor, Tensor]
) -> Tensor:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(rotary_value: Tensor) -> Tensor:
        first, second = rotary_value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    rotary_dim = cos.shape[-1]
    value_rot, value_pass = (value[..., :rotary_dim], value[..., rotary_dim:])
    return torch.cat(
        (value_rot * cos + rotate_half(value_rot) * sin, value_pass), dim=-1
    )


def _apply_rotary(
    query: Tensor, key: Tensor, position_embeddings: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    return (
        _apply_rotary_value(query, position_embeddings),
        _apply_rotary_value(key, position_embeddings),
    )


class EgoBlockQwen35(nn.Module):
    def _project_attention(
        self, attention: nn.Module, hidden: Tensor, *, route: ProjectionRoute
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, sequence_length = hidden.shape[:2]
        head_dim = int(attention.head_dim)
        projected_query = self._project_linear(
            attention.q_proj, hidden, route=route
        ).view(batch_size, sequence_length, -1, head_dim * 2)
        query, gate = projected_query.chunk(2, dim=-1)
        gate = gate.reshape(batch_size, sequence_length, -1)
        query = attention.q_norm(
            query.reshape(batch_size, sequence_length, -1, head_dim)
        ).transpose(1, 2)
        key = attention.k_norm(
            self._project_linear(attention.k_proj, hidden, route=route).view(
                batch_size, sequence_length, -1, head_dim
            )
        ).transpose(1, 2)
        value = (
            self._project_linear(attention.v_proj, hidden, route=route)
            .view(batch_size, sequence_length, -1, head_dim)
            .transpose(1, 2)
        )
        return (query, key, value, gate)

    def __init__(
        self, text_model: nn.Module, config, *, spatial_full=False
    ):
        super().__init__()
        object.__setattr__(self, "_text_model", text_model)
        self.config = config
        self.spatial_full = spatial_full
        self.spatial_rope_spec = SpatialRoPESpec(
            config.spatial_rope_min_one_rad_distance_m,
            config.spatial_rope_max_one_rad_distance_m,
        )
        actual = tuple(
            (
                i
                for i, layer in enumerate(text_model.layers)
                if getattr(layer, "layer_type", getattr(layer, "block_type", None))
                == "full_attention"
            )
        )
        if actual != config.full_attention_layers or not actual:
            raise ValueError("ego blocks require matching native full-attention layers")
        if config.enforce_supported_shape:
            expected = ("linear_attention",) * 3 + ("full_attention",)
            kinds = tuple(
                (
                    getattr(layer, "layer_type", getattr(layer, "block_type", None))
                    for layer in text_model.layers
                )
            )
            if len(kinds) != 32 or kinds != expected * 8:
                raise ValueError(
                    "This model requires 32 layers in 8 groups of 3 GLA + 1 full attention"
                )
        self.local_blocks = nn.ModuleDict()
        self.global_spatial_blocks = nn.ModuleDict()
        for index in actual:
            source = text_model.layers[index]
            if any(
                (
                    not isinstance(getattr(source.self_attn, name), nn.Linear)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            ):
                raise TypeError(
                    "create local blocks from the unwrapped Qwen base before installing routed projections"
                )
            component_names = (
                "input_layernorm",
                "self_attn",
                "post_attention_layernorm",
                "mlp",
            )
            if spatial_full:
                local = FullSpatialModule(source, component_names)
                self.global_spatial_blocks[str(index)] = FullSpatialModule(
                    source, component_names
                )
            else:
                local = nn.Module()
                for name in component_names:
                    local.add_module(name, copy.deepcopy(getattr(source, name)))
                local.requires_grad_(False)
            if not spatial_full:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(local.self_attn, name, LoRALinear(
                        getattr(local.self_attn, name), ("spatial",), config
                    ))
            self.local_blocks[str(index)] = local
        if spatial_full:
            self.spatial_output = FullSpatialModule(text_model, ("norm",))

    def _project_linear(self, projection, hidden, *, route):
        if (
            self.spatial_full
            and route == "spatial"
            and isinstance(projection, SpatialLinear)
        ):
            return projection(hidden)
        if not isinstance(projection, LoRALinear):
            raise TypeError(
                "Expected route-specific projections"
            )
        return projection(hidden, route=route)

    @property
    def text_model(self):
        return object.__getattribute__(self, "_text_model")

    @staticmethod
    def _packed(attention, q, k, v, cu_q, cu_k, max_q, max_k, *, causal):
        out = varlen_attention(
            q[0].transpose(0, 1),
            k[0].transpose(0, 1),
            v[0].transpose(0, 1),
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=float(attention.scaling),
            dropout_p=0.0,
            causal=causal,
        )
        return out[None]

    def _local_layer(self, layer, memory, *, block_cu, max_block, spatial_rope):
        attention = layer.self_attn
        q, k, v, gate = self._project_attention(
            attention, layer.input_layernorm(memory), route="spatial"
        )
        q, k = _apply_rotary(q, k, tuple((t.to(q) for t in spatial_rope)))
        output = self._packed(
            attention, q, k, v, block_cu, block_cu, max_block, max_block, causal=False
        )
        memory = memory + self._project_linear(
            attention.o_proj, output.flatten(2) * gate.sigmoid(), route="spatial"
        )
        return memory + layer.mlp(layer.post_attention_layernorm(memory))

    @staticmethod
    def _trajectory_spans(mask, length):
        if mask is None:
            return ()
        if mask.dtype != torch.bool or mask.shape != (length,):
            raise ValueError(
                "trajectory_mask must be a boolean vector matching prompt length"
            )
        indices = mask.nonzero().flatten().tolist()
        spans = []
        for _, group in itertools.groupby(
            enumerate(indices), lambda item: item[1] - item[0]
        ):
            values = [index for _, index in group]
            if len(values) != 78:
                raise ValueError("each trajectory span must contain exactly 78 queries")
            spans.append((values[0], values[-1] + 1))
        return tuple(spans)

    @staticmethod
    def _route_segments(length, trajectory_spans):
        cursor = 0
        for start, end in trajectory_spans:
            if cursor < start:
                yield (cursor, start, "prompt")
            yield (start, end, "trajectory")
            cursor = end
        if cursor < length:
            yield (cursor, length, "prompt")

    def _prompt_from_prefix(
        self,
        layer,
        prompt,
        key,
        value,
        *,
        prompt_rope,
        image_start,
        image_end,
        prompt_cu,
        joined_cu,
        trajectory_spans=(),
    ):
        attention = layer.self_attn
        segments = tuple(self._route_segments(prompt.shape[1], trajectory_spans))
        normalized = layer.input_layernorm(prompt)
        projections = [
            self._project_attention(attention, normalized[:, start:end], route=route)
            for start, end, route in segments
        ]
        q, k, v = (
            torch.cat([part[i] for part in projections], dim=2) for i in range(3)
        )
        gate = torch.cat([part[3] for part in projections], dim=1)
        q, k = _apply_rotary(q, k, tuple((t.to(q) for t in prompt_rope)))
        if key is None:
            key, value = (k[:, :, :0], v[:, :, :0])
        output = self._packed(
            attention,
            q,
            torch.cat((key, k), dim=2),
            torch.cat((value, v), dim=2),
            prompt_cu,
            joined_cu,
            prompt.shape[1],
            key.shape[2] + prompt.shape[1],
            causal=True,
        )
        if image_end > image_start:
            image = slice(image_start, image_end)
            image_output = dense_attention(
                attention,
                q[:, :, image],
                torch.cat((key, k[:, :, image]), dim=2),
                torch.cat((value, v[:, :, image]), dim=2),
                causal=False,
            )
            output = torch.cat(
                (output[:, :image_start], image_output, output[:, image_end:]), dim=1
            )
        for start, end in trajectory_spans:
            trajectory_output = dense_attention(
                attention,
                q[:, :, start:end],
                torch.cat((key, k[:, :, :end]), dim=2),
                torch.cat((value, v[:, :, :end]), dim=2),
                causal=False,
            )
            output = torch.cat(
                (output[:, :start], trajectory_output, output[:, end:]), dim=1
            )
        gated = output.flatten(2) * gate.sigmoid()
        projected = torch.cat(
            [
                self._project_linear(attention.o_proj, gated[:, start:end], route=route)
                for start, end, route in segments
            ],
            dim=1,
        )
        prompt = prompt + projected
        return prompt + layer.mlp(layer.post_attention_layernorm(prompt))

    def encode_memory(self, memory, *, memory_xyz, block_cu_seqlens, block_bounds=None):
        bounds = block_bounds or offsets(block_cu_seqlens, memory.shape[1], "blocks")
        xyz_ids = memory_xyz.T[:, None]
        spatial_rope = spatial_xyz_position_embeddings(
            memory, xyz_ids, rope_spec=self.spatial_rope_spec
        )
        anchor_rope = self.text_model.rotary_emb(memory, torch.zeros_like(xyz_ids))
        local_kwargs = dict(
            block_cu=block_cu_seqlens.to(dtype=torch.int32),
            max_block=max(b - a for a, b in itertools.pairwise(bounds)),
            spatial_rope=spatial_rope,
        )
        prefixes = []
        for index in self.config.full_attention_layers:
            memory = self._local_layer(
                self.local_blocks[str(index)], memory, **local_kwargs
            )
            layer = (
                self.global_spatial_blocks[str(index)]
                if self.spatial_full
                else self.text_model.layers[index]
            )
            attention = layer.self_attn
            q, k, v, gate = self._project_attention(
                attention, layer.input_layernorm(memory), route="spatial"
            )
            q, k = _apply_rotary(q, k, tuple(t.to(q) for t in anchor_rope))
            out = dense_attention(attention, q, k, v, causal=False)
            memory = memory + self._project_linear(
                attention.o_proj, out.flatten(2) * gate.sigmoid(), route="spatial"
            )
            memory = memory + layer.mlp(layer.post_attention_layernorm(memory))
            prefixes.append((k, v))
        norm = self.spatial_output.norm if self.spatial_full else self.text_model.norm
        return norm(memory), tuple(prefixes)

    def forward_prompt(
        self,
        encoded_memory,
        prompt,
        *,
        position_ids,
        current_image_mask,
        trajectory_mask=None,
        image_bounds=None,
        memory_point_count=None,
    ):
        memory, prefixes = encoded_memory
        if prompt.shape[0] != 1 or position_ids.shape != (3, 1, prompt.shape[1]):
            raise ValueError("Only one unpadded query is supported")
        image_start, image_end = image_bounds or image_span(
            current_image_mask, allow_empty=True
        )
        spans = self._trajectory_spans(trajectory_mask, prompt.shape[1])
        if any(start < image_end and end > image_start for start, end in spans):
            raise ValueError("Trajectory and image spans overlap")
        rope = self.text_model.rotary_emb(prompt, position_ids)
        qcu = torch.tensor(
            [0, prompt.shape[1]], device=prompt.device, dtype=torch.int32
        )
        kcu = torch.tensor(
            [0, memory.shape[1] + prompt.shape[1]],
            device=prompt.device,
            dtype=torch.int32,
        )
        prefix_iter = iter(prefixes)
        for layer in self.text_model.layers:
            kind = getattr(layer, "layer_type", getattr(layer, "block_type", None))
            if kind == "full_attention":
                key, value = next(prefix_iter)
                prompt = self._prompt_from_prefix(
                    layer,
                    prompt,
                    key,
                    value,
                    prompt_rope=rope,
                    image_start=image_start,
                    image_end=image_end,
                    prompt_cu=qcu,
                    joined_cu=kcu,
                    trajectory_spans=spans,
                )
            elif kind == "linear_attention":
                # Native single-sequence GLA: no packing patch or external framework.
                mixed = layer.linear_attn(
                    hidden_states=layer.input_layernorm(prompt),
                    cache_params=None,
                    attention_mask=None,
                )
                prompt = prompt + mixed
                prompt = prompt + layer.mlp(layer.post_attention_layernorm(prompt))
            else:
                raise ValueError("Unsupported layer type: " + str(kind))
        return memory, self.text_model.norm(prompt)
