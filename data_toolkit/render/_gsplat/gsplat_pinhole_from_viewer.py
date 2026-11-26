"""Direct pinhole projection aligned with 3dgs-viewer a0b82a1 (Default/Float).

RGB projection/color math is the validated gsplat_pinhole_from_viewer path.
Depth retains this toolkit's expected Gaussian-center radial-distance contract;
projection-anchor radial distance is used only as the front-to-back sort key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .gsplat_brush_common import (
    ALPHA_CUTOFF, EPS2D, HAS_GSPLAT, SH_INRIA_BIAS, TILE_SIZE,
    brush_sh_to_rgb, c2w_to_w2c, length_robust, quat_to_rotmat, resolve_device,
)

if HAS_GSPLAT:
    import torch
    from gsplat import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles, rasterize_to_pixels
else:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class PinholeIntrinsics:
    fov_x: float
    fov_y: float
    center_uv: tuple[float, float] = (0.5, 0.5)
    skew: float = 0.0  # K[0,1]; zero is exactly the validated viewer model



def fov_to_focal_pinhole(fov: float, pixels: int) -> float:
    return (pixels * 0.5) / math.tan(fov * 0.5)


def _camera_params(width, height, intr):
    if width <= 0 or height <= 0:
        raise ValueError('width and height must be positive')
    if not all(math.isfinite(f) and 0 < f < math.pi for f in (intr.fov_x, intr.fov_y)):
        raise ValueError('pinhole FOV must be finite and in (0, pi) radians')
    if not all(math.isfinite(c) for c in intr.center_uv):
        raise ValueError('center_uv must be finite')
    # Viewer Camera::focal casts f64 FOV-derived focal to f32; center_uv
    # is already f32 before multiplying by the f32 image dimensions.
    return (np.float32(fov_to_focal_pinhole(intr.fov_x, width)),
            np.float32(fov_to_focal_pinhole(intr.fov_y, height)),
            np.float32(intr.center_uv[0]) * np.float32(width),
            np.float32(intr.center_uv[1]) * np.float32(height))


def prepare_native_pinhole_projection(mean_c, scales, rotation_c, width, height, params, support_power, skew=0.0):
    """Port of viewer pinhole.rs::prepare_native_pinhole_projection.

    rotation_c is view_rotation @ normalized_quaternion_rotation. All tensors
    are float32. Returns (projection anchor, opacity multiplier, valid mask).
    """
    fx, fy, cx, cy = params
    qxmin, qxmax = -cx / fx, (width - cx) / fx
    qymin, qymax = -cy / fy, (height - cy) / fy
    planes = mean_c.new_tensor([[1, 0, -qxmin], [-1, 0, qxmax],
                               [0, 1, -qymin], [0, -1, qymax], [0, 0, 1]])
    if skew != 0:
        # Preserve CameraSpec's skewed-K support. At zero skew, leave every
        # operation on the validated viewer path unchanged.
        planes[0, 1] = float(np.float32(skew) / fx)
        planes[1, 1] = float(-np.float32(skew) / fx)
    basis = rotation_c * scales[:, None, :]
    # For each plane: Sigma*n = B*(B^T*n), variance=n^T*Sigma*n.
    local = torch.einsum('nji,pj->npi', basis, planes)
    cov_normals = torch.einsum('nij,npj->npi', basis, local)
    variances = (cov_normals * planes[None]).sum(-1)
    radius = torch.sqrt((2.0 * support_power).clamp_min(0))
    offsets = mean_c.new_tensor([0, 0, 0, 0, -0.01])
    reaches = ((mean_c @ planes.T + offsets + radius[:, None] *
                variances.clamp_min(0).sqrt()) >= 0).all(-1)
    iz = 1.0 / mean_c[:, 2].clamp_min(0.01)
    qx, qy = mean_c[:, 0] * iz, mean_c[:, 1] * iz
    if skew != 0:
        qx = qx + (np.float32(skew) / fx) * qy
    inside = ((mean_c[:, 2] > 0.01) & (qx >= qxmin) & (qx < qxmax) &
              (qy >= qymin) & (qy < qymax))
    depth_std = variances[:, 4].clamp_min(0).sqrt()
    extent = radius * depth_std
    front = mean_c[:, 2] - extent > 0.01
    anchor = mean_c.clone()
    for _ in range(2):
        for p in range(4):
            distance = (anchor * planes[p]).sum(-1)
            step = torch.where(distance < 0, -distance / variances[:, p].clamp_min(1e-12), 0)
            anchor = anchor + cov_normals[:, p] * step[:, None]
    anchor = torch.where(inside[:, None], mean_c, anchor)
    tol = 1e-5 * length_robust(anchor).clamp_min(1)
    anchor_inside = ((anchor @ planes[:4].T) >= -tol[:, None]).all(-1)
    stable = ((anchor[:, 2] > 0.01) & (anchor[:, 2] - extent > 0.01) &
              (extent / anchor[:, 2].clamp_min(0.01) <= 0.33))
    delta_local = torch.einsum('nji,nj->ni', rotation_c, anchor - mean_c)
    distance_sq = (delta_local / scales.clamp_min(1e-7)).square().sum(-1)
    tail = torch.exp(-0.5 * distance_sq)
    rescue_base = reaches & anchor_inside & (distance_sq <= 2 * support_power)
    stable_rescue = rescue_base & front & stable
    dual_rescue = rescue_base & ~front & ~stable
    center_excess = (3 * depth_std / anchor[:, 2].clamp_min(0.01) / 0.33 - 1).clamp_min(0)
    center_gain = torch.exp(-0.1 * center_excess)
    gain = torch.where(inside, center_gain, torch.where(dual_rescue, tail * 0.45, tail))
    valid = ((inside | stable_rescue | dual_rescue) & torch.isfinite(anchor).all(-1) &
             torch.isfinite(gain))
    return anchor, gain, valid


def _empty_view(means):
    return {key: means.new_zeros(shape, dtype=dtype) for key, shape, dtype in [
        ('means2d', (0, 2), torch.float32), ('radii', (0, 2), torch.int32),
        ('depths', (0,), torch.float32), ('ranges', (0,), torch.float32), ('conics', (0, 3), torch.float32),
        ('colors', (0, 3), torch.float32), ('opacities', (0,), torch.float32)]}


def _project_pinhole_view(
    torch_splats: dict,
    c2w: np.ndarray,
    width: int,
    height: int,
    intrinsics: PinholeIntrinsics,
    *,
    sh_bias: float,
) -> dict:
    splats = torch_splats
    means = splats['means']
    params = _camera_params(width, height, intrinsics)
    fx, fy, cx, cy = params
    view = torch.as_tensor(c2w_to_w2c(c2w), device=means.device, dtype=torch.float32)
    mean_c = means @ view[:3, :3].T + view[:3, 3]
    quats, scales, opacity = splats['quats'], splats['scales'], splats['opacities']
    qnorm_sq = quats.square().sum(-1)
    keep = (torch.isfinite(mean_c).all(-1) & (mean_c[:, 2] <= 1e10) &
            (length_robust(mean_c) >= 0.01) & torch.isfinite(scales).all(-1) &
            torch.isfinite(qnorm_sq) & (qnorm_sq >= 1e-6) &
            torch.isfinite(opacity) & (opacity >= ALPHA_CUTOFF))
    means, mean_c, scales, quats, opacity, sh = [x[keep] for x in
        (means, mean_c, scales, quats, opacity, splats['sh'])]
    if not len(means):
        return _empty_view(means)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    rotation_c = view[:3, :3][None] @ quat_to_rotmat(quats)
    anchor, gain, valid = prepare_native_pinhole_projection(
        mean_c, scales, rotation_c, width, height, params, torch.log(opacity * 255), intrinsics.skew)
    ranges = length_robust(mean_c[valid])
    means, scales, opacity, sh, rotation_c, anchor, gain = [x[valid] for x in
        (means, scales, opacity, sh, rotation_c, anchor, gain)]
    if not len(means):
        return _empty_view(means)
    iz = 1.0 / anchor[:, 2]
    means2d = torch.stack((fx * anchor[:, 0] * iz + cx, fy * anchor[:, 1] * iz + cy), -1)
    if intrinsics.skew != 0:
        means2d[:, 0] += np.float32(intrinsics.skew) * anchor[:, 1] * iz
    fw, fh = np.float32(width), np.float32(height)
    xz = (anchor[:, 0] * iz).clamp((np.float32(-0.15) * fw - cx) / fx,
                                  (np.float32(1.15) * fw - cx) / fx)
    yz = (anchor[:, 1] * iz).clamp((np.float32(-0.15) * fh - cy) / fy,
                                  (np.float32(1.15) * fh - cy) / fy)
    zero = torch.zeros_like(iz)
    J = torch.stack((torch.stack((fx * iz, zero, -fx * iz * xz), -1),
                     torch.stack((zero, fy * iz, -fy * iz * yz), -1)), -2)
    if intrinsics.skew != 0:
        sx = np.float32(intrinsics.skew)
        image_qx = (anchor[:, 0] + (sx / fx) * anchor[:, 1]) * iz
        clamped = image_qx.clamp((np.float32(-0.15) * fw - cx) / fx,
                                 (np.float32(1.15) * fw - cx) / fx)
        J[:, 0, 1] = sx * iz
        J[:, 0, 2] = -fx * iz * clamped
    V = J @ (rotation_c * scales[:, None, :])
    cov = V @ V.transpose(-1, -2)
    max_abs = cov.abs().amax((-2, -1))
    cov = cov * torch.where(max_abs > 1e18, 1e18 / max_abs, 1)[:, None, None]
    cov[:, 0, 0] += EPS2D
    cov[:, 1, 1] += EPS2D
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * c - b * b
    inv_det = torch.where(det > 0, 1 / det, 0)
    conic = torch.stack((c * inv_det, -b * inv_det, a * inv_det), -1)
    opacity = opacity * gain
    power = torch.log(opacity * 255)
    det_conic = conic[:, 0] * conic[:, 2] - conic[:, 1].square()
    inv_conic_det = torch.where(det_conic > 0, 1 / det_conic, 0)
    ex = (2 * power * conic[:, 2] * inv_conic_det).sqrt()
    ey = (2 * power * conic[:, 0] * inv_conic_det).sqrt()
    keep = (torch.isfinite(cov).all((-2, -1)) & (opacity >= ALPHA_CUTOFF) &
            (det_conic > 0) & (ex >= 0) & (ey >= 0) & torch.isfinite(ex) & torch.isfinite(ey) &
            torch.isfinite(means2d).all(-1) &
            (means2d[:, 0] + ex > 0) & (means2d[:, 0] - ex < width) &
            (means2d[:, 1] + ey > 0) & (means2d[:, 1] - ey < height))
    cam_pos = torch.as_tensor(c2w[:3, 3], device=means.device, dtype=torch.float32)
    dirs = means[keep] - cam_pos
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    colors = brush_sh_to_rgb(dirs, sh[keep]) + sh_bias
    # Viewer finalizes SH+bias at [-100,100], then raster max(color,0).
    # gsplat accumulates colors verbatim: apply that raster clamp here.
    colors = torch.where(torch.isfinite(colors), colors, 0).clamp(0, 100)
    # Anchor radial distance is the latest native pinhole sort key.
    # Clip integer radii only at a screen-covering radius to avoid int overflow.
    max_radius = 2 * max(width, height) + 2
    radii = torch.stack((ex[keep], ey[keep]), -1).ceil().clamp(max=max_radius).to(torch.int32)
    return dict(means2d=means2d[keep], radii=radii, depths=length_robust(anchor[keep]),
                conics=conic[keep], colors=colors, opacities=opacity[keep], ranges=ranges[keep])


def float_rgb_tensor_to_u8(rgb: "torch.Tensor") -> np.ndarray:
    """Viewer Float RGB8 conversion: nonnegative round-half-up."""
    values = rgb.detach().cpu().numpy()
    return np.floor(np.clip(values, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def _rasterize_views_padded(
    views: list[dict],
    width: int,
    height: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Render RGB, premultiplied radial distance, and alpha together."""
    C = len(views)
    if C == 0:
        return []
    device = views[0]["means2d"].device
    max_n = max(int(v["means2d"].shape[0]) for v in views)
    if max_n == 0:
        return [
            (
                np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.float32),
                np.zeros((height, width), dtype=np.float32),
            )
            for _ in range(C)
        ]

    means2d = torch.zeros((C, max_n, 2), device=device, dtype=torch.float32)
    radii = torch.zeros((C, max_n, 2), device=device, dtype=torch.int32)
    depths = torch.zeros((C, max_n), device=device, dtype=torch.float32)
    ranges = torch.zeros((C, max_n), device=device, dtype=torch.float32)
    conics = torch.zeros((C, max_n, 3), device=device, dtype=torch.float32)
    colors = torch.zeros((C, max_n, 3), device=device, dtype=torch.float32)
    opacities = torch.zeros((C, max_n), device=device, dtype=torch.float32)

    for i, v in enumerate(views):
        n = int(v["means2d"].shape[0])
        if n == 0:
            continue
        means2d[i, :n] = v["means2d"]
        radii[i, :n] = v["radii"]
        depths[i, :n] = v["depths"]
        ranges[i, :n] = v["ranges"]
        conics[i, :n] = v["conics"]
        colors[i, :n] = v["colors"]
        opacities[i, :n] = v["opacities"]

    tile_width = int(math.ceil(width / float(TILE_SIZE)))
    tile_height = int(math.ceil(height / float(TILE_SIZE)))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        TILE_SIZE,
        tile_width,
        tile_height,
        packed=False,
        n_images=C,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    renders, alpha = rasterize_to_pixels(
        means2d,
        conics,
        torch.cat([colors, ranges[..., None]], dim=-1),
        opacities,
        width,
        height,
        TILE_SIZE,
        isect_offsets,
        flatten_ids,
        packed=False,
    )
    return [
        (
            float_rgb_tensor_to_u8(renders[i, ..., :3]),
            renders[i, ..., 3].cpu().numpy(),
            alpha[i, ..., 0].cpu().numpy(),
        )
        for i in range(C)
    ]


def render_pinhole_batch(
    splats: dict,
    c2ws: Sequence[np.ndarray],
    width: int,
    height: int,
    intrinsics: PinholeIntrinsics,
    *,
    sh_bias: float = SH_INRIA_BIAS,
    device: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pinhole RGB8, premultiplied radial distance, and alpha; one raster pass."""
    if not HAS_GSPLAT:
        raise RuntimeError("torch/gsplat required for gsplat_pinhole_from_viewer")
    if not c2ws:
        return []
    device = resolve_device(device)
    with torch.inference_mode():
        views = [
            _project_pinhole_view(
                splats, c2w, width, height, intrinsics, sh_bias=sh_bias
            )
            for c2w in c2ws
        ]
        return _rasterize_views_padded(views, width, height)
