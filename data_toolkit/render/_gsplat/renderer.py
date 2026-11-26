"""Direct Brush-aligned pinhole RGB, expected radial distance and alpha."""

import math
import numpy as np
from ..geometry import rigid_transform
from .gsplat_pinhole_from_viewer import PinholeIntrinsics, render_pinhole_batch


class CalibratedPinhole:
    """Render directly at the caller's output calibration and resolution."""

    def __init__(self, k, width, height):
        from ..camera import CameraSpec

        k = CameraSpec(width, height, k).K
        self.width, self.height = width, height
        self.intrinsics = PinholeIntrinsics(
            fov_x=2 * math.atan(width * 0.5 / k[0, 0]),
            fov_y=2 * math.atan(height * 0.5 / k[1, 1]),
            center_uv=(k[0, 2] / width, k[1, 2] / height),
            skew=float(k[0, 1]),
        )

    def render(self, splats, camera_c2w_cv, device):
        pose = rigid_transform(camera_c2w_cv)
        rgb, weighted_range, alpha = render_pinhole_batch(
            splats, [pose], self.width, self.height, self.intrinsics, device=device,
        )[0]
        distance = np.divide(
            weighted_range, alpha, out=np.full_like(alpha, np.inf), where=alpha > 1e-8
        )
        return rgb, distance, alpha
