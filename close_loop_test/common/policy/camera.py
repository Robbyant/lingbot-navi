import math
import numpy as np
import quaternion

def _habitat_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = float(width) / 2.0 / np.tan(np.deg2rad(float(hfov_deg)) / 2.0)
    return np.array([[fx, 0.0, float(width) / 2.0], [0.0, fx, float(height) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def _habitat_pose_to_opencv_c2w(sensor_xyz, sensor_quat) -> np.ndarray:
    """Simulator camera pose as a LingBot OpenCV camera-to-world matrix."""
    rotation_h = _quat_xyzw_matrix(sensor_quat)
    translation_h = np.asarray(sensor_xyz, dtype=np.float64).reshape(3)
    basis = _OPENCV_FROM_HABITAT
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = basis @ rotation_h @ basis
    c2w[:3, 3] = basis @ translation_h
    return c2w

_OPENCV_FROM_HABITAT = np.diag([1., -1., -1.])
def _quat_xyzw_matrix(xyzw) -> np.ndarray:
    (x, y, z, w) = (float(value) for value in xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=np.float64)
