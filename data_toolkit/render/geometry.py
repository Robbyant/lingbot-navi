"""Coordinate transforms for offline renders (asset world: Z up, cameras: OpenCV)."""

import numpy as np

# Columns are OpenCV right/down/forward axes expressed in FLU coordinates.
FLU_FROM_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def points(value, width=3):
    a = np.asarray(value, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, width))
    if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
        raise ValueError(f"Expected finite [N,{width}] array")
    return a


def transform_points(xyz, transform):
    return points(xyz) @ transform[:3, :3].T + transform[:3, 3]


def rigid_transform(value):
    t = np.asarray(value, dtype=np.float64)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Expected a rigid, right-handed 4x4 transform")
    return t


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Expected a nonzero finite xyzw quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def look_at(position, target):
    position, target = np.asarray(position), np.asarray(target)
    forward = target - position
    distance = np.linalg.norm(forward)
    if distance < 1e-6:
        raise ValueError("Camera and target coincide")
    forward = forward / distance
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    t = np.eye(4)
    t[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    t[:3, 3] = position
    return t


def bev_camera(bounds, z, size):
    """Equal metric scale; +X right, +Y up. Return C2W, K and actual bounds."""
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("BEV bounds must be xmin ymin xmax ymax")
    cx, cy = (b[:2] + b[2:]) / 2
    span = max(b[2] - b[0], b[3] - b[1])
    b = np.array([cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2])
    t = np.eye(4)
    t[:3, :3] = np.diag([1.0, -1.0, -1.0])
    t[:3, 3] = [cx, cy, z]
    scale = (size - 1) / span
    k = np.array(
        [[scale, 0.0, (size - 1) / 2], [0.0, scale, (size - 1) / 2], [0.0, 0.0, 1.0]]
    )
    return t, k, b


def bev_pixels(xyz, bounds, size):
    xyz = points(xyz)
    scale = (size - 1) / (bounds[2] - bounds[0])
    return np.column_stack(
        ((xyz[:, 0] - bounds[0]) * scale, (bounds[3] - xyz[:, 1]) * scale)
    )


def pinhole_pixels(xyz, camera, k):
    eye = (points(xyz) - camera[:3, 3]) @ camera[:3, :3]
    z = eye[:, 2]
    uvw = eye @ k.T
    return uvw[:, :2] / np.maximum(z[:, None], 1e-8), z
