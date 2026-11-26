import math
import numpy as np

def relative_execution_path(points, predicted_start_pose, actual_xy, actual_heading):
    """Anchor predicted FLU motion once at execution time (ground robot: yaw only).

    predicted_start_pose is world xyz + xyzw, with +X forward in Robby FLU.
    Simulator pose only places the relative controls; it never repairs localization.
    """
    path = np.asarray(points, dtype=np.float64)
    pose = np.asarray(predicted_start_pose, dtype=np.float64)
    actual = np.asarray(actual_xy, dtype=np.float64)
    heading = np.asarray(actual_heading, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 2 or len(path) == 0 or (pose.shape != (7,)) or (actual.shape != (2,)) or (heading.shape != (2,)) or (not all((np.isfinite(v).all() for v in (path, pose, actual, heading)))):
        raise ValueError('缺少或无效的模型预测起始位姿/轨迹，请重新规划')
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-08 or np.linalg.norm(heading) < 1e-08:
        raise ValueError('起始朝向无效，请重新规划')
    (x, y, z, w) = pose[3:] / norm
    forward = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    if np.linalg.norm(forward) < 1e-08:
        raise ValueError('预测朝向没有有效的水平分量，请重新规划')
    forward /= np.linalg.norm(forward)
    heading = heading / np.linalg.norm(heading)
    predicted_rotation = np.array([[forward[0], -forward[1]], [forward[1], forward[0]]])
    actual_rotation = np.array([[heading[0], -heading[1]], [heading[1], heading[0]]])
    return (path[:, :2] - pose[:2]) @ predicted_rotation @ actual_rotation.T + actual

def simplify_follow_polyline(points, gap: float=0.2) -> np.ndarray:
    """Drop points closer than ``gap`` and always keep the end of the path."""
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2 or len(raw) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xy = raw[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    kept = [xy[0]]
    for point in xy[1:-1]:
        if float(np.linalg.norm(point - kept[-1])) >= gap:
            kept.append(point)
    if len(xy) > 1 and float(np.linalg.norm(xy[-1] - kept[-1])) > 1e-06:
        kept.append(xy[-1])
    return np.asarray(kept, dtype=np.float64)

def habitat_heading_model_xy(rotation_xyzw) -> list[float]:
    """Habitat body forward in robby XY: (habitat_x, -habitat_z)."""
    (qx, qy, qz, qw) = (float(v) for v in rotation_xyzw)
    (vx, vy, vz) = (0.0, 0.0, -1.0)
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz
    hx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    hz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    heading = np.asarray([hx, -hz], dtype=np.float64)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-09:
        return [1.0, 0.0]
    heading /= norm
    return [float(heading[0]), float(heading[1])]

def rank_retrieval_centers(centers, counts, rotation, origin, previous_xy):
    """Keep replans nearest the previous world-XY goal, independent of density."""
    world = np.asarray(centers) @ np.asarray(rotation).T + np.asarray(origin)
    distances = np.linalg.norm(world[:, :2] - previous_xy, axis=1) if previous_xy is not None else None

    def priority(i):
        if distances is not None:
            return (float(distances[i]), i)
        return (-(counts[i] if counts is not None else 0), i)
    return (sorted(range(len(world)), key=priority), world, distances)
