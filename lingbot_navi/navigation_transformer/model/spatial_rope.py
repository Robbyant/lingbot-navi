"""Metric XYZ rotary position embeddings for scene memory attention."""

import torch

from ...spatial_encoding import DEFAULT_SPATIAL_ROPE_SPEC, SpatialRoPESpec


@torch.no_grad()
def spatial_xyz_position_embeddings(
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    rope_spec: SpatialRoPESpec = DEFAULT_SPATIAL_ROPE_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legacy metric XYZ tables without consulting Qwen MRoPE.

    The spatial specification owns its width, axis ordering and frequencies.
    Anchor/image/text positions remain the native Qwen module's responsibility.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("Spatial XYZ position_ids must be [3, B, S]")
    pair_count = rope_spec.rotary_dim // 2
    xyz = position_ids.permute(1, 2, 0).float().to(hidden.device)
    frequencies = rope_spec.xyz_phases(xyz).flatten(-2)
    remainder = pair_count - frequencies.shape[-1]
    if remainder:
        frequencies = torch.nn.functional.pad(frequencies, (0, remainder))
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos()
    sin = embedding.sin()
    return (cos.to(dtype=hidden.dtype), sin.to(dtype=hidden.dtype))
