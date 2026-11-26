# Core extracted from vln_data b582bf13 / the mesh-free odometry demo.
"""Shared Brush-aligned 3DGS helpers for direct pinhole capture.

SH, length_robust, quaternion math, covariance blur constants, and gsplat availability.
Projection math stays in the pinhole module.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from gsplat import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles, rasterize_to_pixels

    HAS_GSPLAT = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    isect_offset_encode = None  # type: ignore
    isect_tiles = None  # type: ignore
    rasterize_to_pixels = None  # type: ignore
    HAS_GSPLAT = False

# Brush / Inria capture defaults
EPS2D = 0.3
ALPHA_CUTOFF = 1.0 / 255.0
TILE_SIZE = 16  # Brush TILE_WIDTH
SH_INRIA_BIAS = 0.5  # Inria PLY: f_dc=(rgb-0.5)/C0; needed for rgb8 match


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w).astype(np.float32)


def length_robust(v: "torch.Tensor") -> "torch.Tensor":
    """Brush Vec3A::length_robust (overflow-safe L2)."""
    scale = v.abs().amax(dim=-1)
    nonzero = scale > 0
    safe = torch.where(nonzero, scale, torch.ones_like(scale))
    s = v / safe.unsqueeze(-1)
    return torch.where(nonzero, scale * s.norm(dim=-1), torch.zeros_like(scale))


def quat_to_rotmat(quats: "torch.Tensor") -> "torch.Tensor":
    """quats (N,4) as (w,x,y,z) → (N,3,3), matching Brush Quat::to_mat3."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def brush_sh_to_rgb(dirs: "torch.Tensor", sh: "torch.Tensor") -> "torch.Tensor":
    """Match brush `sh_coeffs_to_color` (no +0.5). dirs (N,3) unit, sh (N,16,3)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    c0 = 0.2820948
    color = sh[:, 0, :] * c0
    f0a = 0.4886025
    color = color + sh[:, 1, :] * (-f0a * y).unsqueeze(-1)
    color = color + sh[:, 2, :] * (f0a * z).unsqueeze(-1)
    color = color + sh[:, 3, :] * (-f0a * x).unsqueeze(-1)
    z2 = z * z
    f0b = -1.0925485 * z
    f1a = 0.54627424
    fc1 = x * x - y * y
    fs1 = 2.0 * x * y
    color = color + sh[:, 4, :] * (f1a * fs1).unsqueeze(-1)
    color = color + sh[:, 5, :] * (f0b * y).unsqueeze(-1)
    color = color + sh[:, 6, :] * (0.9461747 * z2 - 0.31539157).unsqueeze(-1)
    color = color + sh[:, 7, :] * (f0b * x).unsqueeze(-1)
    color = color + sh[:, 8, :] * (f1a * fc1).unsqueeze(-1)
    f0c = -2.285229 * z2 + 0.4570458
    f1b = 1.4453057 * z
    f2a = -0.5900436
    fc2 = x * fc1 - y * fs1
    fs2 = x * fs1 + y * fc1
    color = color + sh[:, 9, :] * (f2a * fs2).unsqueeze(-1)
    color = color + sh[:, 10, :] * (f1b * fs1).unsqueeze(-1)
    color = color + sh[:, 11, :] * (f0c * y).unsqueeze(-1)
    color = color + sh[:, 12, :] * (z * (1.8658817 * z2 - 1.119529)).unsqueeze(-1)
    color = color + sh[:, 13, :] * (f0c * x).unsqueeze(-1)
    color = color + sh[:, 14, :] * (f1b * fc1).unsqueeze(-1)
    color = color + sh[:, 15, :] * (f2a * fc2).unsqueeze(-1)
    return color.clamp(-100.0, 100.0)


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if not HAS_GSPLAT:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def float_rgb_to_u8(rgb: np.ndarray) -> np.ndarray:
    """Brush-style round(clip·255) → uint8."""
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def float_rgb_tensor_to_u8(rgb: "torch.Tensor") -> np.ndarray:
    """Same rounding as ``float_rgb_to_u8``, from a torch float RGB tensor."""
    return float_rgb_to_u8(rgb.detach().cpu().numpy())
