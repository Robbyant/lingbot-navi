"""Pinhole output calibration, independent of capture height and rendering device."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    K: np.ndarray

    def __post_init__(self):
        k = np.asarray(self.K, dtype=np.float64)
        if (
            any(
                type(side) is not int or side <= 0 for side in (self.width, self.height)
            )
            or k.shape != (3, 3)
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[1, 0]) > 1e-8
            or not 0 <= k[0, 2] < self.width
            or not 0 <= k[1, 2] < self.height
        ):
            raise ValueError("Invalid pinhole image dimensions or calibration")
        k = k.copy()
        k.flags.writeable = False
        object.__setattr__(self, "K", k)

    @classmethod
    def from_yaml(cls, path=None):
        """Load an explicit calibration, defaulting to the bundled 640x480 camera."""
        import yaml

        path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("configs") / "lingbot_pinhole.yaml"
        )
        data = yaml.safe_load(path.read_text())
        return cls(width=data["width"], height=data["height"], K=data["K"])

    @classmethod
    def from_hfov(cls, width=640, height=480, hfov_deg=120.0):
        if not math.isfinite(hfov_deg) or not 0 < hfov_deg < 180:
            raise ValueError(
                "Horizontal field of view must lie strictly between 0 and 180"
            )
        focal = width * 0.5 / math.tan(math.radians(hfov_deg) * 0.5)
        return cls(
            width,
            height,
            np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
        )

    @property
    def hfov_deg(self):
        """Horizontal ray span through the principal row, including off-center K."""
        return math.degrees(
            math.atan(self.K[0, 2] / self.K[0, 0])
            + math.atan((self.width - self.K[0, 2]) / self.K[0, 0])
        )

    def resized(self, width, height):
        k = self.K.copy()
        k[0] *= width / self.width
        k[1] *= height / self.height
        return CameraSpec(width, height, k)

    def cropped(self, left, top, width, height):
        if any(type(v) is not int for v in (left, top, width, height)) or (
            min(left, top) < 0
            or left + width > self.width
            or top + height > self.height
        ):
            raise ValueError("Crop must fit inside the image")
        k = self.K.copy()
        k[0, 2] -= left
        k[1, 2] -= top
        return CameraSpec(width, height, k)
