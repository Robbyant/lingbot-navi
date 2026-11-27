"""Shared perspective capture and orthographic BEV, without model weights."""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image


PERSPECTIVE_RENDERER = "brush_aligned_native_pinhole_gsplat_v3"


@dataclass(frozen=True)
class RenderResult:
    """RGB image, metric expected depth and accumulated opacity.

    Perspective depth is radial range, with infinity at unsupported pixels;
    orthographic depth is axial. Alpha is coverage, not a collision mask.
    """

    rgb: Image.Image
    depth: np.ndarray
    alpha: np.ndarray
    depth_type: str


def require_backend(device):
    import torch

    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError(
            "Offline rendering needs: pip install -r requirements.txt"
        ) from exc
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Offline Gaussian rendering requires a CUDA device")
    return torch, rasterization


class GaussianScene:
    def __init__(self, path, device="cuda:0", *, backend="auto"):
        if backend not in ("auto", "fused_cuda", "gsplat"):
            raise ValueError(f"Unsupported backend {backend!r}; expected 'auto', 'fused_cuda', or 'gsplat'")
        torch, rasterization = require_backend(device)
        self.torch, self.rasterization, self.device = torch, rasterization, device
        self.backend = backend
        path = Path(path)
        if path.suffix.lower() != ".ply":
            raise ValueError("Expected a standard Gaussian .ply file")
        from .ply import read_gaussian_ply

        self.xyz, quats, scales, opacity, colors, self.sh_degree = read_gaussian_ply(
            path
        )
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        if (
            not np.isfinite(self.xyz).all()
            or not np.isfinite(scales).all()
            or (scales <= 0).any()
            or (norms < 1e-6).any()
            or not np.isfinite(quats).all()
            or not np.isfinite(opacity).all()
            or not np.isfinite(colors).all()
        ):
            raise ValueError("Invalid Gaussian positions, scales, or rotations")
        quats /= norms
        # Pad lower SH degrees for the shared Brush-aligned color evaluator.
        if colors.shape[1] < 16:
            colors = np.pad(colors, ((0, 0), (0, 16 - colors.shape[1]), (0, 0)))

        self._host_arrays = (self.xyz, quats, scales, opacity, colors)
        self._fused_scene = None
        self.tensors = None

        if backend in ("auto", "fused_cuda"):
            from ._gsplat.fused_brush_cuda import FusedBrushScene, is_available

            if is_available():
                self._fused_scene = FusedBrushScene(
                    self.xyz, quats, scales, opacity, colors, device=device
                )
            elif backend == "fused_cuda":
                raise RuntimeError(
                    "Requested backend='fused_cuda' but CUDA JIT toolchain (nvcc/ninja) is unavailable."
                )

        if self._fused_scene is None:
            self.tensors = [torch.tensor(x, device=device) for x in self._host_arrays]

        self._calibration_key = None
        self._calibration = None

    def _ensure_gsplat_tensors(self):
        if self.tensors is None:
            self.tensors = [
                self.torch.tensor(x, device=self.device) for x in self._host_arrays
            ]
        return self.tensors

    def close(self):
        if self._fused_scene is not None:
            self._fused_scene.close()
            self._fused_scene = None
        self.tensors = None
        self._host_arrays = None
        self._calibration = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render_camera(self, camera, camera_to_world_cv):
        """Render a CameraSpec at an OpenCV camera-to-asset-world pose (meters)."""
        return self.render(camera_to_world_cv, camera.K, camera.width, camera.height)

    def render(self, camera, k, width, height, *, ortho=False, z_range=None):
        """Return RenderResult. Perspective depth is radial distance.

        Perspective RGB uses the same calibrated path for memory capture, query
        synthesis, grounding and localization. BEV retains orthographic gsplat.
        """
        if not ortho and z_range is None and self._fused_scene is not None:
            rgb, distance, alpha = self._fused_scene.render(camera, k, width, height)
            return RenderResult(Image.fromarray(rgb), distance, alpha, "radial")

        torch = self.torch
        args = self._ensure_gsplat_tensors()
        if z_range is not None:
            # Center-based cutaway; this does not analytically truncate ellipsoids.
            keep = (self.xyz[:, 2] >= z_range[0]) & (self.xyz[:, 2] <= z_range[1])
            if not keep.any():
                raise ValueError("No Gaussian centers inside the requested Z interval")
            mask = torch.tensor(keep, device=self.device)
            args = [x[mask] for x in args]
        if not ortho:
            from ._gsplat.renderer import CalibratedPinhole

            key = (int(width), int(height), tuple(np.asarray(k).reshape(-1)))
            if key != self._calibration_key:
                self._calibration = CalibratedPinhole(k, width, height)
                self._calibration_key = key
            splats = dict(zip(("means", "quats", "scales", "opacities", "sh"), args))
            rgb, distance, alpha = self._calibration.render(splats, camera, self.device)
            return RenderResult(Image.fromarray(rgb), distance, alpha, "radial")
        with torch.inference_mode():
            rgbd, alpha, _ = self.rasterization(
                *args,
                torch.tensor(
                    np.linalg.inv(camera), device=self.device, dtype=torch.float32
                )[None],
                torch.tensor(k, device=self.device, dtype=torch.float32)[None],
                width,
                height,
                packed=True,
                sh_degree=self.sh_degree,
                near_plane=0.04,
                far_plane=100,
                radius_clip=0.0,
                camera_model="ortho",
                render_mode="RGB+ED",
            )
            rgbd, alpha = rgbd[0].cpu().numpy(), alpha[0, :, :, 0].cpu().numpy()
        rgb = rgbd[:, :, :3] + (1 - alpha[:, :, None]) * np.array([0.95, 0.95, 0.93])
        return RenderResult(
            Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)),
            rgbd[:, :, 3],
            alpha,
            "axial",
        )
