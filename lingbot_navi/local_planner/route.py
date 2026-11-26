"""External ordered world-XYZ route and current-ego route window."""
import numpy as np

def rotation_matrix(quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError("rotation must be a finite unit XYZW quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def world_to_ego(points, position, rotation_xyzw):
    return ((np.asarray(points, dtype=np.float64) - np.asarray(position)) @ rotation_matrix(rotation_xyzw)).astype(np.float32)

def polyline(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("path must contain finite XYZ")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=-1) > 1e-8]
    points = points[keep]
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
    return points, distances


def interpolate(points, distances, at):
    return np.stack([np.interp(at, distances, points[:, axis]) for axis in range(3)], -1).astype(np.float32)

def route_condition_input(route_world, current_position, current_rotation, config):
    points, arc = polyline(world_to_ego(route_world, current_position, current_rotation))
    if len(points) == 1:
        matched = 0.
        offset = float(np.linalg.norm(points[0]))
    else:
        delta = np.diff(points, axis=0)
        alpha = np.clip((-points[:-1] * delta).sum(1) / (delta * delta).sum(1), 0., 1.)
        projected = points[:-1] + alpha[:, None] * delta
        distances = np.linalg.norm(projected, axis=1)
        index = int(np.argmin(distances))
        matched = float(arc[index] + alpha[index] * (arc[index + 1] - arc[index]))
        offset = float(distances[index])
    start = max(0., matched - config.route_crop_behind_m)
    end = min(float(arc[-1]), matched + config.route_crop_ahead_m)
    supplied = interpolate(points, arc, np.linspace(start, end, config.global_points))
    return supplied, dict(policy='ego_projected_arc_window_v1', points=config.global_points,
                         match_arc_m=matched, match_offset_m=offset, start_arc_m=start,
                         end_arc_m=end, retained_behind_m=matched-start,
                         retained_ahead_m=end-matched, source_length_m=float(arc[-1]))
