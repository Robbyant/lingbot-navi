"""Explicit Habitat Y-up <-> memory Z-up and FLU camera transforms."""
import numpy as np

ZUP_FROM_HABITAT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
# Habitat camera: right +X, up +Y, forward -Z. Memory camera: FLU.
HABITAT_CAMERA_FROM_FLU = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])




def transform_points(points, matrix):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite Nx3 points')
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def camera_intrinsics(width, height, hfov):
    focal = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
