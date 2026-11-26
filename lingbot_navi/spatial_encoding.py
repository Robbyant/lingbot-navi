"""Metric coordinate spectrum shared by the scene and local planners."""

from dataclasses import dataclass
import math
from typing import ClassVar
import torch
from torch import Tensor


@dataclass(frozen=True)
class SpatialRoPESpec:
    profile: ClassVar[str] = "metric_xyz_legacy_v1"
    rotary_dim: ClassVar[int] = 64
    frequencies_per_axis: ClassVar[int] = 10
    pair_layout: ClassVar[str] = "split_half"
    axis_layout: ClassVar[str] = "frequency_major_xyz"
    min_one_rad_distance_m: float = 0.01
    max_one_rad_distance_m: float = 1000.0

    def __post_init__(self):
        lo, hi = (self.min_one_rad_distance_m, self.max_one_rad_distance_m)
        if not (math.isfinite(lo) and math.isfinite(hi) and (0 < lo < hi)):
            raise ValueError(
                "spatial RoPE distances must be finite and satisfy 0 < min < max"
            )

    def inverse_frequencies(self, device):
        return torch.logspace(
            math.log10(self.min_one_rad_distance_m),
            math.log10(self.max_one_rad_distance_m),
            steps=self.frequencies_per_axis,
            device=device,
            dtype=torch.float32,
        ).reciprocal()

    def xyz_phases(self, xyz_m: torch.Tensor) -> torch.Tensor:
        """Shared differentiable FP32 phases for all metric coordinate encoders."""
        if xyz_m.ndim < 1 or xyz_m.shape[-1] != 3:
            raise ValueError("metric XYZ must end with three coordinates")
        if not xyz_m.is_floating_point():
            raise TypeError("metric XYZ must be floating point")
        return (
            xyz_m.float()[..., None, :]
            * self.inverse_frequencies(xyz_m.device)[:, None]
        )


DEFAULT_SPATIAL_ROPE_SPEC = SpatialRoPESpec()


METRIC_FREQUENCY_COUNT = SpatialRoPESpec.frequencies_per_axis
METRIC_FOURIER_DIM = 3 * 2 * METRIC_FREQUENCY_COUNT


def metric_xyz_phases(
    xyz_m: Tensor, *, spec: SpatialRoPESpec = DEFAULT_SPATIAL_ROPE_SPEC
) -> Tensor:
    return spec.xyz_phases(xyz_m)


def metric_xyz_fourier_features(
    xyz_m: Tensor, *, spec: SpatialRoPESpec = DEFAULT_SPATIAL_ROPE_SPEC
) -> Tensor:
    """Return the shared sin/cos features, ordered by frequency then XYZ."""
    phases = metric_xyz_phases(xyz_m, spec=spec).flatten(-2)
    return torch.cat((phases.sin(), phases.cos()), dim=-1)
