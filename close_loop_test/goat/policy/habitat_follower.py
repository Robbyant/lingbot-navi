"""Navmesh-assisted execution of model waypoints; never reads benchmark goals."""
import math
import numpy as np
import habitat_sim
HEIGHT_EPS_M = 1e-6
from .navmesh_motion import compatible_navmesh_start, START_VERTICAL_TOL_M

def closest_triangle_points(triangles, target):
    """Closest XZ point on each triangle, with its interpolated floor height."""
    t = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    q = np.asarray(target, dtype=np.float64)[[0, 2]]
    (a, b, c) = (t[:, 0], t[:, 1], t[:, 2])
    (ab, ac) = ((b - a)[:, [0, 2]], (c - a)[:, [0, 2]])
    aq = q - a[:, [0, 2]]
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    den = cross(ab, ac)
    valid = np.abs(den) > 1e-12
    divisor = np.where(valid, den, 1.0)
    (u, v) = (cross(aq, ac) / divisor, cross(ab, aq) / divisor)
    inside = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    candidates = []
    for (start, end) in ((a, b), (b, c), (c, a)):
        edge = (end - start)[:, [0, 2]]
        alpha = np.clip(np.sum((q - start[:, [0, 2]]) * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-12), 0.0, 1.0)
        candidates.append(start + alpha[:, None] * (end - start))
    edges = np.stack(candidates, axis=1)
    distances = np.sum((edges[:, :, [0, 2]] - q) ** 2, axis=2)
    result = edges[np.arange(len(t)), np.argmin(distances, axis=1)]
    result[inside] = (a + u[:, None] * (b - a) + v[:, None] * (c - a))[inside]
    return (result, valid)

class HabitatWaypointFollower:

    def __init__(self, sim, *, goal_radius=0.2, max_projection_m=2.0, max_height_delta_m=0.5):
        values = (goal_radius, max_projection_m, max_height_delta_m)
        if not all((math.isfinite(float(v)) and float(v) > 0 for v in values)):
            raise ValueError('Habitat follower distances must be finite and positive')
        self.sim = sim
        self.agent = sim.get_agent(0)
        self.pf = sim.pathfinder
        self.goal_radius = float(goal_radius)
        self.max_projection_m = float(max_projection_m)
        self.max_height_delta_m = float(max_height_delta_m)
        self.follower = sim.make_greedy_follower(agent_id=0, goal_radius=self.goal_radius, stop_key=None, forward_key='move_forward', left_key='turn_left', right_key='turn_right')
        self._triangles = {}
        self._reference = None
        self._target = None
        self.index = 0
        self.diagnostics = {}
        self.route = []
        self.status = ''

    def _path(self, start, end):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if not self.pf.find_path(path) or not math.isfinite(path.geodesic_distance):
            return None
        return path

    def _project(self, goal, start):
        island = self.pf.get_island(start)
        if island < 0 or not self.pf.is_navigable(start):
            height_limit = START_VERTICAL_TOL_M
            try:
                (reference, _) = compatible_navmesh_start(self.pf, start, max_vertical_m=height_limit)
            except ValueError as error:
                raise ValueError('robot_off_navmesh') from error
            island = self.pf.get_island(reference)
            if island < 0:
                raise ValueError('robot_off_navmesh')
        if island not in self._triangles:
            vertices = np.asarray(self.pf.build_navmesh_vertices(island), dtype=np.float64)
            self._triangles[island] = vertices.reshape(-1, 3, 3)
        triangles = self._triangles[island]
        if not len(triangles):
            raise ValueError('empty_robot_island')
        (candidates, valid) = closest_triangle_points(triangles, goal)
        reference_y = start[1]
        height_limit = self.max_height_delta_m
        valid &= np.abs(candidates[:, 1] - reference_y) <= height_limit + HEIGHT_EPS_M
        distance = np.linalg.norm(candidates[:, [0, 2]] - goal[[0, 2]], axis=1)
        distance[~valid] = np.inf
        for i in np.argsort(distance):
            if distance[i] > self.max_projection_m:
                break
            point = candidates[i].copy()
            toward = triangles[i].mean(axis=0) - point
            length = np.linalg.norm(toward[[0, 2]])
            if length > 1e-09:
                point += toward * min(1.0, 0.0001 / length)
            snapped = np.asarray(self.pf.snap_point(point, island), dtype=np.float64)
            if not np.isfinite(snapped).all() or np.linalg.norm(snapped - point) > 0.01 or abs(snapped[1] - reference_y) > height_limit + HEIGHT_EPS_M:
                continue
            gap = float(np.linalg.norm(snapped[[0, 2]] - goal[[0, 2]]))
            if gap > self.max_projection_m:
                continue
            path = self._path(start, snapped)
            if path is not None:
                return (snapped, gap, island, path)
        raise ValueError('no_reachable_projection_within_limits')

    def project_route(self, goal_xy):
        """Preview executor reachability without advancing the agent or follower."""
        start = np.asarray(self.agent.get_state().position, dtype=np.float64)
        y = start[1]
        goal = np.asarray(goal_xy, dtype=np.float64)
        try:
            (target, gap, island, path) = self._project(np.array([goal[0], y, -goal[1]]), start)
            return ([[float(p[0]), -float(p[2])] for p in path.points], dict(projection_distance_m=gap, island=island, geodesic_m=float(path.geodesic_distance)))
        except ValueError as error:
            return ([], dict(error=str(error)))

    def navigation_backtrack(self, polyline):
        """Original endpoint failed: probe at 1 m intervals of reverse path length."""
        points = np.asarray(polyline, dtype=np.float64)[::-1]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        detail = dict(original_endpoint_xy=points[0].tolist(), probes=[])
        for distance in range(1, int(math.ceil(arc[-1]))):
            j = int(np.searchsorted(arc, distance, side='right')) - 1
            point = points[j] + (distance - arc[j]) / lengths[j] * (points[j + 1] - points[j])
            (route, info) = self.project_route(point)
            detail['probes'].append(dict(backtrack_m=distance, candidate_xy=point.tolist(), **info))
            if route and info['geodesic_m'] > self.goal_radius:
                detail.update(backtrack_m=distance, selected_endpoint_xy=route[-1])
                return (point, detail)
        return (None, detail)

    def exploration_endpoint(self, polyline, *, step_m=0.25):
        """Probe the model route backwards for a reachable, nontrivial goal.

        Probes never move the agent. The normal projection, island and floor
        constraints apply to every candidate, including the original endpoint.
        """
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or (not len(points)) or (not np.isfinite(points).all()):
            raise ValueError('model path must be finite Nx2')
        if not math.isfinite(float(step_m)) or step_m <= 0:
            raise ValueError('exploration probe step must be finite and positive')
        probes = 0
        rejected = {}
        backtrack = 0.0
        last_candidate = None

        def probe(candidate, distance):
            nonlocal probes, last_candidate
            if last_candidate is not None and np.array_equal(candidate, last_candidate):
                return None
            last_candidate = candidate.copy()
            probes += 1
            (route, detail) = self.project_route(candidate)
            error = detail.get('error')
            if not route:
                error = error or 'empty_projected_route'
            elif detail['geodesic_m'] <= self.goal_radius:
                error = 'candidate_already_reached'
            if error:
                rejected[error] = rejected.get(error, 0) + 1
                return None
            return (np.asarray(route[-1], dtype=np.float64), dict(original_endpoint_xy=points[-1].tolist(), candidate_xy=candidate.tolist(), selected_endpoint_xy=route[-1], backtrack_m=float(distance), probe_count=probes, rejected=dict(rejected), **detail))
        for index in range(len(points) - 1, 0, -1):
            (end, start) = (points[index], points[index - 1])
            length = float(np.linalg.norm(end - start))
            count = max(1, int(math.ceil(length / step_m)))
            for offset in range(count):
                distance = min(offset * step_m, length)
                candidate = end + (start - end) * (distance / length) if length else end
                result = probe(candidate, backtrack + distance)
                if result is not None:
                    return result
            backtrack += length
        result = probe(points[0], backtrack)
        if result is not None:
            return result
        return (None, dict(error='no_reachable_exploration_endpoint', original_endpoint_xy=points[-1].tolist(), backtrack_m=backtrack, probe_count=probes, rejected=rejected))

    def next_action(self, polyline):
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or (not len(points)) or (not np.isfinite(points).all()):
            raise ValueError('model path must be finite Nx2')
        if self._reference is None or not np.array_equal(points, self._reference):
            self._reference = points.copy()
            self.index = 1 if len(points) > 1 else 0
            self._target = None
            self.follower.reset()
        position = np.asarray(self.agent.get_state().position, dtype=np.float64)
        for _ in range(len(points) + 1):
            reference_y = position[1]
            requested = np.array([points[self.index, 0], reference_y, -points[self.index, 1]])
            self.diagnostics = dict(executor='habitat', uses_ground_truth_navmesh=True, waypoint_index=self.index, waypoint_count=len(points), requested_goal_xyz=requested.tolist(), goal_radius_m=self.goal_radius, max_projection_m=self.max_projection_m, max_height_delta_m=self.max_height_delta_m)
            try:
                if self._target is None:
                    (target, gap, island, path) = self._project(requested, position)
                    self._target = (target, gap, island)
                else:
                    (target, gap, island) = self._target
                    path = self._path(position, target)
                    if path is None:
                        raise ValueError('waypoint_no_longer_reachable')
                self.route = [[float(p[0]), -float(p[2])] for p in path.points]
                self.diagnostics.update(projected_goal_xyz=target.tolist(), projection_distance_m=gap, substituted=gap > 0.05, island=island, remaining_geodesic_m=float(path.geodesic_distance))
                if path.geodesic_distance <= self.goal_radius:
                    if self.index < len(points) - 1:
                        self.index += 1
                        self._target = None
                        self.follower.reset()
                        continue
                    self.status = '已到可达替代终点' if gap > 0.05 else '已到轨迹终点'
                    self.diagnostics['completion'] = 'projected_endpoint' if gap > 0.05 else 'model_endpoint'
                    return (None, self.status)
                was_planning = getattr(self.sim, '_navmesh_follower_planning', False)
                self.sim._navmesh_follower_planning = True
                try:
                    action = self.follower.next_action_along(np.asarray(target, dtype=np.float32))
                finally:
                    self.sim._navmesh_follower_planning = was_planning
                if action is None:
                    raise ValueError('follower_stopped_before_waypoint')
                if action not in ('move_forward', 'turn_left', 'turn_right'):
                    raise ValueError('unexpected_follower_action')
                self.status = f'Habitat 航点 {self.index + 1}/{len(points)} · 剩余 {path.geodesic_distance:.2f} m'
                self.diagnostics['action'] = action
                return (action, '')
            except (ValueError, habitat_sim.errors.GreedyFollowerError) as error:
                self.diagnostics['error'] = str(error) or type(error).__name__
                self.status = 'Habitat 路径执行失败：' + self.diagnostics['error']
                return (None, self.status)
        raise RuntimeError('waypoint iteration did not terminate')
