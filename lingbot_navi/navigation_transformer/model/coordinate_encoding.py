"""Metric coordinate features, spatial embeddings and trajectory order encoding."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from ...spatial_encoding import (
    DEFAULT_SPATIAL_ROPE_SPEC,
    METRIC_FOURIER_DIM,
    SpatialRoPESpec,
    metric_xyz_fourier_features,
)


def ordered_query_progress_encoding(query_count: int, hidden_dim: int) -> Tensor:
    """Return fixed sinusoidal order features with no coordinate content."""
    position = torch.arange(query_count, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )
    encoding = torch.zeros(query_count, hidden_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding


class SpatialPositionEncoder(nn.Module):
    """Shared ungated centroid-relative XYZ embedding for Spatial and endpoint tokens."""

    def __init__(
        self,
        hidden_dim: int,
        position_hidden_dim: int = 384,
        *,
        coordinate_spec: SpatialRoPESpec = DEFAULT_SPATIAL_ROPE_SPEC,
    ) -> None:
        super().__init__()
        self.coordinate_spec = coordinate_spec
        self.raw_projection = nn.Linear(3, hidden_dim)
        self.fourier_projection = nn.Sequential(
            nn.Linear(METRIC_FOURIER_DIM, position_hidden_dim),
            nn.GELU(),
            nn.Linear(position_hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, xyz_m: Tensor) -> Tensor:
        if xyz_m.ndim < 2 or xyz_m.shape[-1] != 3:
            raise ValueError("XYZ in metres must have shape [..., 3]")
        xyz_m = xyz_m.to(device=self.raw_projection.weight.device)
        fourier = metric_xyz_fourier_features(xyz_m, spec=self.coordinate_spec)
        encoded = self.output_norm(
            self.raw_projection(xyz_m.to(self.raw_projection.weight))
            + self.fourier_projection(fourier.to(self.raw_projection.weight))
        )
        return encoded
