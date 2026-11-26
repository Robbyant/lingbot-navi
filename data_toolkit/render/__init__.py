"""Shared 3D Gaussian rendering. GPU dependencies load when a scene is opened."""

from .camera import CameraSpec
from .renderer import GaussianScene, PERSPECTIVE_RENDERER, RenderResult, require_backend

__all__ = [
    "CameraSpec",
    "GaussianScene",
    "PERSPECTIVE_RENDERER",
    "RenderResult",
    "require_backend",
]
