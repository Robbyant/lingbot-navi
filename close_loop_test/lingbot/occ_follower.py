"""Follow a model or exploration endpoint using the pinned OCC action grid."""
import heapq
import itertools
import math

import numpy as np


class OccupancyWaypointFollower:
    def __init__(self, sim, pose, *, forward, turn_deg, goal_radius=.20, max_projection_m=2.):
        self.sim = sim
        self.pose = pose
        self.navigation = sim.occupancy_navigation
        self.forward = forward
        self.turn = math.radians(turn_deg)
        self.goal_radius = goal_radius
        self.max_projection_m = max_projection_m
        self.route = []
        self.status = ''
        self.diagnostics = {}
        self._reference = None
        self._field = None

    def _candidate(self, start, goal):
        target = self.navigation.project(start, goal, self.max_projection_m)
        if target is None:
            return None, None
        return target, self.navigation.field([self.navigation.cell(target)])

    def project_route(self, goal):
        start, _ = self.pose()
        target, field = self._candidate(start, goal)
        distance = self.navigation.distance(start, field)
        if distance is None:
            return [], {'error': 'no_reachable_projection_within_limits'}
        return self.navigation.route(start, target, field), {
            'projection_distance_m': float(np.linalg.norm(target - goal)),
            'geodesic_m': distance, 'distance_metric': 'occ_grid_8_neighbor_v1'}

    def _backtrack(self, polyline, spacing, include_endpoint):
        points = np.asarray(polyline, float)[::-1]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arcs = np.r_[0., np.cumsum(lengths)]
        distances = np.arange(0. if include_endpoint else spacing, arcs[-1], spacing)
        if include_endpoint:
            distances = np.r_[distances, arcs[-1]]
        detail = {'original_endpoint_xy': points[0].tolist(), 'probes': []}
        for distance in distances:
            index = np.searchsorted(arcs, distance, side='right') - 1
            candidate = points[index]
            if index < len(lengths):
                candidate = candidate + (distance - arcs[index]) / lengths[index] * (points[index+1] - candidate)
            route, info = self.project_route(candidate)
            detail['probes'].append(dict(backtrack_m=float(distance), candidate_xy=candidate.tolist(), **info))
            if route and info['geodesic_m'] > self.goal_radius:
                target = np.asarray(route[-1])
                return target, {**detail, **info, 'backtrack_m': float(distance),
                                'selected_endpoint_xy': target.tolist()}
        return None, detail

    def navigation_backtrack(self, polyline):
        return self._backtrack(polyline, 1., False)

    def exploration_endpoint(self, polyline, *, step_m=.25):
        return self._backtrack(polyline, step_m, True)

    def _action(self, start, heading):
        """Bounded search over the simulator's real turn/forward primitives."""
        angle = math.atan2(heading[1], heading[0])
        turns = max(1, round(2 * math.pi / self.turn))
        unit = self.navigation.resolution / 4
        order = itertools.count()

        def key(xy, yaw):
            return tuple(np.rint(xy / unit).astype(int)) + (yaw % turns,)

        def potential(xy):
            distance = self.navigation.distance(xy, self._field)
            return (math.inf if distance is None else
                    max(distance, float(np.linalg.norm(xy - self._target)) - self.goal_radius / 2))

        initial = potential(start)
        queue = [(initial, next(order), 0., start, 0, 0, None)]
        visited = {key(start, 0): 0.}
        best = None
        best_value = initial
        for _ in range(2500):
            if not queue:
                break
            _, _, cost, point, yaw, depth, first = heapq.heappop(queue)
            if cost > visited[key(point, yaw)]:
                continue
            if np.linalg.norm(point - self._target) <= self.goal_radius:
                return first
            value = potential(point)
            if value < best_value - .03:
                best, best_value = first, value
                if initial - value >= 1.:
                    break
            if depth >= 20:
                continue
            for action, offset in (('move_forward', 0), ('turn_left', 1), ('turn_right', -1)):
                candidate = point
                if not offset:
                    theta = angle + yaw * self.turn
                    candidate = point + self.forward * np.array([math.cos(theta), math.sin(theta)])
                    if not self.navigation.segment_allowed(point, candidate):
                        continue
                new_cost = cost + (.055 if offset else self.forward)
                new_yaw = yaw + offset
                state = key(candidate, new_yaw)
                heuristic = potential(candidate)
                if not math.isfinite(heuristic) or new_cost >= visited.get(state, math.inf):
                    continue
                visited[state] = new_cost
                heapq.heappush(queue, (new_cost + 1.05 * heuristic, next(order), new_cost,
                                       candidate, new_yaw, depth+1, first or action))
        return best

    def next_action(self, polyline):
        start, heading = self.pose()
        start = np.asarray(start, dtype=float)
        goal = np.asarray(polyline, float)[-1]
        if self._reference is None or not np.array_equal(self._reference, goal):
            self._reference = goal.copy()
            self._target, self._field = self._candidate(start, goal)
        remaining = self.navigation.distance(start, self._field)
        self.diagnostics = {'executor': 'occ', 'requested_goal_xy': goal.tolist(),
                            'uses_ground_truth_navmesh': False}
        if remaining is None:
            self.route = []
            self.diagnostics['error'] = 'no_reachable_projection_within_limits'
            return None, 'OCC 终点不可达'
        self.route = self.navigation.route(start, self._target, self._field)
        gap = float(np.linalg.norm(self._target - goal))
        self.diagnostics.update(effective_goal_xy=self._target.tolist(), projection_distance_m=gap,
                                remaining_geodesic_m=remaining,
                                distance_metric='occ_grid_8_neighbor_v1', waypoint_index=0)
        if np.linalg.norm(start - self._target) <= self.goal_radius:
            self.diagnostics['completion'] = 'projected_endpoint' if gap > 1e-6 else 'model_endpoint'
            self.status = '已到可达替代终点' if gap > 1e-6 else '已到轨迹终点'
            return None, self.status
        action = self._action(start, heading)
        if action is None:
            self.diagnostics['error'] = 'occ_no_progress'
            return None, 'OCC 内没有可推进的离散动作'
        self.status = 'OCC 终点导航'
        return action, ''


class ExplorationMotionGuard:
    """Bound stationary scans and commit to one OCC-safe escape direction."""

    def __init__(self, navigation, forward, turn_deg, start):
        self.navigation = navigation
        self.forward = forward
        self.turn = math.radians(turn_deg)
        self.turn_limit = math.ceil(360 / turn_deg)
        self.anchor = np.asarray(start, dtype=float)
        self.visited = {navigation.cell(self.anchor)}
        self.stationary_steps = 0
        self.escape = None
        self.escape_remaining = 0

    def observe(self, position):
        xy = np.asarray(position, dtype=float)
        if np.linalg.norm(xy - self.anchor) >= self.forward / 2:
            self.anchor = xy
            self.visited.add(self.navigation.cell(xy))
            self.stationary_steps = 0
            if self.escape is not None:
                self.escape_remaining -= 1
                if self.escape_remaining <= 0:
                    self.escape = None
        else:
            self.stationary_steps += 1

    def action(self, position, heading):
        """Return (override, action, newly_committed)."""
        if self.escape is None and self.stationary_steps < self.turn_limit:
            return False, None, False
        xy = np.asarray(position, dtype=float)
        angle = math.atan2(heading[1], heading[0])
        fresh = self.escape is None
        if fresh:
            candidates = []
            for turn in range(self.turn_limit):
                signed = turn if 2 * turn <= self.turn_limit else turn - self.turn_limit
                yaw = angle + signed * self.turn
                direction = np.array([math.cos(yaw), math.sin(yaw)])
                safe_steps = 0
                for count in range(1, 5):
                    if not self.navigation.segment_allowed(xy, xy + direction * self.forward * count):
                        break
                    safe_steps = count
                if safe_steps:
                    destination = self.navigation.cell(xy + direction * self.forward)
                    candidates.append((destination not in self.visited, safe_steps, -abs(signed), signed, direction))
            if not candidates:
                return True, None, True
            _, self.escape_remaining, _, _, self.escape = max(candidates, key=lambda row: row[:4])
        if self.stationary_steps >= 2 * self.turn_limit:
            return True, None, fresh
        # With a fixed direction the robot spends at most half a turn aligning.
        desired = math.atan2(self.escape[1], self.escape[0])
        delta = math.atan2(math.sin(desired - angle), math.cos(desired - angle))
        if abs(delta) > self.turn / 2:
            return True, 'turn_left' if delta > 0 else 'turn_right', fresh
        direction = np.array([math.cos(angle), math.sin(angle)])
        if not self.navigation.segment_allowed(xy, xy + direction * self.forward):
            return True, None, fresh
        return True, 'move_forward', fresh
