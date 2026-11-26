"""GOAT memory camera conversion matching the qwen G01 arithmetic.

The optical quaternion round trip is intentional: an algebraically equivalent
matrix conversion changes float32 positions and downstream model predictions.
"""
import math
import numpy as np

_HABITAT_TO_ZUP_MATRIX = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]], dtype=np.float64)
_R_WH = _HABITAT_TO_ZUP_MATRIX

def _quat_normalize_xyzw(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / n, y / n, z / n, w / n

def _quat_to_matrix_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )

def _matrix_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize_xyzw(x, y, z, w)

def habitat_position_to_zup(position_xyz: list[float] | tuple[float, ...]) -> list[float]:
    """Map habitat world position (Y-up) to Z-up export world (Z vertical)."""
    x, y, z = (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]))
    p = _HABITAT_TO_ZUP_MATRIX @ np.array([x, y, z], dtype=np.float64)
    return [float(p[0]), float(p[1]), float(p[2])]

def habitat_camera_pose_to_zup(
    position_xyz: list[float] | tuple[float, ...],
    rotation_xyzw: list[float] | tuple[float, ...],
) -> tuple[list[float], list[float]]:
    """Convert habitat camera pose to Z-up export with a right-handed optical frame.

    Habitat/OpenGL camera: +X right, +Y up, optical axis = -Z (still right-handed).
    Export camera: +X right, +Y down, +Z forward (standard right-handed optical frame).
    """
    pos_z = habitat_position_to_zup(position_xyz)

    hx, hy, hz, hw = (float(v) for v in rotation_xyzw)
    rot_h = _quat_to_matrix_xyzw(hx, hy, hz, hw)
    # Transform all three camera axes. Reconstructing axes from forward alone
    # preserves ordinary yaw/pitch but removes roll and is singular vertically.
    # Habitat OpenGL -> OpenCV: right stays +X; up/forward change sign.
    rot_z = _R_WH @ rot_h @ np.diag([1.0, -1.0, -1.0])
    ox, oy, oz, ow = _matrix_to_quat_xyzw(rot_z)
    return pos_z, [ox, oy, oz, ow]

def habitat_agent_pose_to_zup_camera(
    agent_position_xyz: list[float] | tuple[float, ...],
    rotation_xyzw: list[float] | tuple[float, ...],
    cam_height_m: float,
) -> tuple[list[float], list[float]]:
    """Habitat agent foot pose -> Z-up camera optical-center pose (+Z forward)."""
    x, y, z = (float(agent_position_xyz[0]), float(agent_position_xyz[1]), float(agent_position_xyz[2]))
    cam_hab = [x, y + float(cam_height_m), z]
    return habitat_camera_pose_to_zup(cam_hab, rotation_xyzw)

def simulator_camera(agent_xyz, sensor_quat, cam_height):
    position, rotation = habitat_agent_pose_to_zup_camera(agent_xyz, sensor_quat, cam_height)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = _quat_to_matrix_xyzw(*rotation)
    c2w[:3, 3] = position
    basis = np.eye(4, dtype=np.float64)
    basis[:3, :3] = [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]
    # Match the offline einsum before the float32 conversion in the pipeline.
    return np.einsum('nij,jk->nik', c2w[None], basis)[0].astype(np.float32)
