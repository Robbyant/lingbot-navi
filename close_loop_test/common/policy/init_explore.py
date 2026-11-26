"""Depth-based frontier exploration with Habitat waypoint execution.

Build a 2D occupancy grid, rank frontier viewpoints by visible unknown area,
and issue navigation targets or observation turns. Habitat handles translation.
Failed targets are cooled down before selecting another viewpoint. Low coverage
gain triggers retargeting; revisited routes can revive unapproached frontiers.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from .camera import _habitat_K, _habitat_pose_to_opencv_c2w
from .directional_collision import DirectionalCollision, remember_contact
UNKNOWN = np.int8(0)
FREE = np.int8(1)
OCCUPIED = np.int8(2)
CELL_M = 0.1
MAX_RANGE_M = 4.0
ROBOT_RADIUS_M = 0.17
PLAN_BUFFER_M = 0.025
COMFORT_BUFFER_M = PLAN_BUFFER_M
VIEW_STANDOFF_M = 1.5
FRONTIER_OBSERVATION_M = 2.35
COLLISION_TTL_STEPS = 24
OBSTACLE_MIN_M = 0.25
OBSTACLE_MAX_M = 1.5
MIN_CLUSTER = 3
ALIGN_RAD = math.radians(35.0)
LOOKAHEAD_M = 0.75

def pose_xy_heading(sensor_xyz, sensor_quat) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and horizontal heading in the follow plane."""
    c2w = _habitat_pose_to_opencv_c2w(sensor_xyz, sensor_quat)
    forward = c2w[:3, 2]
    heading = np.array([forward[0], forward[2]], dtype=np.float64)
    scale = float(np.linalg.norm(heading))
    if scale < 1e-09:
        heading = np.array([1.0, 0.0], dtype=np.float64)
    else:
        heading = heading / scale
    xy = np.array([c2w[0, 3], c2w[2, 3]], dtype=np.float64)
    return (xy, heading)

def _unit(heading) -> np.ndarray:
    vector = np.asarray(heading, dtype=np.float64).reshape(2)
    scale = float(np.linalg.norm(vector))
    if scale < 1e-09:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vector / scale

def _rotate(heading: np.ndarray, radians: float) -> np.ndarray:
    """Counterclockwise is a left turn in the follow plane."""
    cosine = math.cos(radians)
    sine = math.sin(radians)
    (x, y) = heading
    return np.array([cosine * x - sine * y, sine * x + cosine * y], dtype=np.float64)

class ExploreMap:
    """Occupancy, viewpoint coverage, and one-step frontier exploration."""

    def __init__(self, *, cell_m: float=CELL_M, max_range_m: float=MAX_RANGE_M, robot_radius_m: float=ROBOT_RADIUS_M, hfov_deg: float=42.0, forward_m: float=0.25, turn_deg: float=30.0, visibility_height: tuple[float, float] | None=None, obstacle_height: tuple[float, float]=(OBSTACLE_MIN_M, OBSTACLE_MAX_M)) -> None:
        self.obstacle_height = obstacle_height
        self.visibility_map = ExploreMap(cell_m=cell_m, max_range_m=max_range_m, robot_radius_m=robot_radius_m, hfov_deg=hfov_deg, forward_m=forward_m, turn_deg=turn_deg, obstacle_height=visibility_height) if visibility_height is not None else None
        self.cell_m = float(cell_m)
        self.max_range_m = float(max_range_m)
        self.robot_radius_m = float(robot_radius_m)
        self.hfov_deg = float(hfov_deg)
        self.forward_m = float(forward_m)
        self.turn_rad = math.radians(float(turn_deg))
        self.plan_clearance_m = float(robot_radius_m) + PLAN_BUFFER_M
        self.comfort_clearance_m = float(robot_radius_m) + COMFORT_BUFFER_M
        self.origin: np.ndarray | None = None
        self._collision_patches: list[tuple[DirectionalCollision, int]] = []
        self._collision_memory: list[DirectionalCollision] = []
        self._sticky_centroid: np.ndarray | None = None
        self._committed_goal: np.ndarray | None = None
        self._external_observation = None
        self._retry_frontiers = []
        self._external_observe_anchor = None
        self._external_observe_turns = 0
        self._view_history: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_goal: np.ndarray | None = None
        self._frontier_clock = 0
        self.stop_reason = ''
        self.decisions = []
        self._decision_step = 0
        self._failed_viewpoints = []
        self._failed_frontiers = []
        self._coverage_history = []
        self._motion_history = []
        self._near_frontier_mode = False
        self.recovery_info = None
        self._approach_standoff = VIEW_STANDOFF_M
        self._scan_turns = 0
        self.plan_xy: list[np.ndarray] = []
        self.goal_xy: np.ndarray | None = None
        self.occ = np.zeros((0, 0), dtype=np.int8)
        self.viewed = np.zeros((0, 0), dtype=bool)
        self.frontier_seen = np.zeros((0, 0), dtype=bool)
        self.edge_frontier = np.zeros((0, 0), dtype=bool)
        self._frontier_born = np.zeros((0, 0), dtype=np.int32)

    def integrate_depth(self, depth, sensor_xyz, sensor_quat, feet_y: float, *, width: int, height: int, hfov_deg: float | None=None, _projected=None) -> None:
        """Raycast one depth frame into the grid.

        The 0.25–1.5 m body layer above the feet includes low obstacles. A hit is an
        obstacle and the ray up to it is free. Other heights are ignored, so they
        do not create free space or frontiers. A column with no point in the layer
        stays unknown. On the left and right edges of the field of view, the free
        cells closer than an obstacle stay frontiers.
        """
        metric = np.asarray(depth, dtype=np.float32)
        if metric.ndim == 3 and metric.shape[-1] == 1:
            metric = metric[..., 0]
        if metric.ndim != 2:
            raise ValueError(f'depth must be HxW, got {metric.shape}')
        if metric.shape != (height, width):
            (height, width) = metric.shape
        hfov = self.hfov_deg if hfov_deg is None else float(hfov_deg)
        intrinsics = _habitat_K(width, height, hfov)
        c2w = _habitat_pose_to_opencv_c2w(sensor_xyz, sensor_quat)
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        camera_xy = np.array([translation[0], translation[2]], dtype=np.float64)
        self._mark_disk(camera_xy, self.robot_radius_m + self.cell_m, FREE)
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        surface = getattr(self, '_visibility_surface', False)
        column_stride = max(1, min(4, int(round(self.cell_m / 0.02)))) if surface else 4
        row_stride = 8
        surface_rays = []
        if _projected is None and self.visibility_map is not None:
            _projected = {}
        edge_columns = {0, (width - 1) // column_stride * column_stride}
        edge_cells: set[tuple[int, int]] = set()
        for u in range(0, width, column_stride):
            nearest_obstacle = None
            farthest_clear = None
            for v in range(0, height, row_stride):
                depth_m = float(metric[v, u])
                if not math.isfinite(depth_m) or depth_m <= 0.0001:
                    continue
                key = (hfov, u, v)
                world = None if _projected is None else _projected.get(key)
                if world is None:
                    camera = np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=np.float64)
                    world = rotation @ camera + translation
                    if _projected is not None:
                        _projected[key] = world
                habitat_y = -float(world[1])
                height_m = habitat_y - float(feet_y)
                if not self.obstacle_height[0] <= height_m <= self.obstacle_height[1]:
                    continue
                point_xy = np.array([world[0], world[2]], dtype=np.float64)
                delta = point_xy - camera_xy
                distance = float(np.linalg.norm(delta))
                if distance < 0.0001:
                    continue
                if distance > self.max_range_m:
                    direction = delta / distance
                    clipped = camera_xy + direction * self.max_range_m
                    if farthest_clear is None or self.max_range_m > farthest_clear[0]:
                        farthest_clear = (self.max_range_m, clipped)
                    continue
                if nearest_obstacle is None or distance < nearest_obstacle[0]:
                    nearest_obstacle = (distance, point_xy)
                if farthest_clear is None or distance > farthest_clear[0]:
                    farthest_clear = (distance, point_xy)
            if nearest_obstacle is not None:
                free_cells = self._mark_ray(camera_xy, nearest_obstacle[1], occupied_end=True)
            elif farthest_clear is not None:
                free_cells = self._mark_ray(camera_xy, farthest_clear[1], occupied_end=False)
            else:
                free_cells = []
            if surface:
                evidence = nearest_obstacle or farthest_clear
                surface_rays.append(None if evidence is None else evidence[1].copy())
            if u in edge_columns:
                edge_cells.update((cell for cell in free_cells if self._touches_unknown(*cell, diagonal=True)))
        if surface:
            self._fill_visibility_surface(camera_xy, surface_rays)
        self._commit_edge_frontier(edge_cells)
        if self.visibility_map is not None:
            v = self.visibility_map
            v._visibility_surface = True
            v.integrate_depth(depth, sensor_xyz, sensor_quat, feet_y, width=width, height=height, hfov_deg=hfov_deg, _projected=_projected)
            self._prepare(v.origin)
            self._prepare(v.origin + (np.array(v.occ.shape[::-1]) - 1) * v.cell_m)

    def _fill_visibility_surface(self, origin, rays):
        """Join adjacent valid pixel rays only across continuous surfaces.

        This layer is visibility evidence only, never collision evidence.
        Missing returns and depth jumps remain unknown.
        """
        fill = np.zeros(self.occ.shape, dtype=np.uint8)
        for (left, right) in zip(rays, rays[1:]):
            if left is None or right is None:
                continue
            (dl, dr) = (np.linalg.norm(left - origin), np.linalg.norm(right - origin))
            surface_inset_m = 0.04
            if min(dl, dr) <= surface_inset_m or abs(dl - dr) > 0.1:
                continue
            a = origin + (left - origin) * ((dl - surface_inset_m) / dl)
            b = origin + (right - origin) * ((dr - surface_inset_m) / dr)
            polygon = np.array([self._index(origin), self._index(a), self._index(b)], np.int32)
            cv2.fillConvexPoly(fill, polygon, 1)
        free = (fill != 0) & (self.occ != OCCUPIED)
        self.occ[free] = FREE
        self.viewed[free] = True

    def _visibility_grid(self, field='occ'):
        """Align visibility evidence without changing the collision grid."""
        v = self.visibility_map
        if v is None:
            return getattr(self, field)
        source = getattr(v, field)
        result = np.zeros(self.occ.shape, dtype=source.dtype)
        if v.origin is None:
            return result
        offset = np.rint((v.origin - self.origin) / self.cell_m).astype(int)
        (x0, y0) = np.maximum(offset, 0)
        (x1, y1) = np.minimum(offset + np.array(v.occ.shape[::-1]), np.array(self.occ.shape[::-1]))
        if x1 > x0 and y1 > y0:
            result[y0:y1, x0:x1] = source[y0 - offset[1]:y1 - offset[1], x0 - offset[0]:x1 - offset[0]]
        return result

    def _visibility_occluded(self, origin, target):
        return (self.visibility_map or self)._occluded(origin, target)

    def mark_blocked_ahead(self, xy, heading) -> None:
        """Remember inferred contact ahead without covering our own body."""
        heading = _unit(heading)
        radius = min(self.cell_m, self.forward_m * 0.25)
        contact = remember_contact(self._collision_memory, xy, heading, self.robot_radius_m, radius)
        self._collision_patches.append((contact, COLLISION_TTL_STEPS))
        # Match LingBot-Navi's collision retargeting with Habitat execution.
        self.decisions.append(dict(event='escape', reason='collision',
            old_goal=None if self._committed_goal is None else self._committed_goal.tolist()))
        if self._committed_goal is not None:
            self._failed_viewpoints.append((self._committed_goal.copy(), self._decision_step + 24))
        if self._sticky_centroid is not None:
            self._failed_frontiers.append((self._sticky_centroid.copy(), self._decision_step + 24))
        self._committed_goal = None
        self._sticky_centroid = None

    def confirm_footprint(self, xy):
        """Reconcile only the body disk at a pose actually occupied by the agent.

        This is stronger evidence than a ray's FREE cells: normal ray painting
        intentionally cannot erase OCCUPIED cells. Never use it at a prediction.
        """
        xy = np.asarray(xy, dtype=float)
        self._frontier_robot_xy = xy.copy()
        self._prepare(xy)
        (x, y) = self._index(xy)
        reach = int(math.ceil(self.robot_radius_m / self.cell_m)) + 1
        corrected = 0
        for iy in range(max(0, y - reach), min(self.occ.shape[0], y + reach + 1)):
            for ix in range(max(0, x - reach), min(self.occ.shape[1], x + reach + 1)):
                if np.linalg.norm(self._center(ix, iy) - xy) >= self.robot_radius_m:
                    continue
                corrected += int(self.occ[iy, ix] == OCCUPIED)
                self.occ[iy, ix] = FREE
                self.viewed[iy, ix] = True
        return corrected

    def mark_traversed(self, start, end) -> None:
        """Successful body motion can disprove inferred contacts along its sweep.

        Measured depth occupancy is never erased by this feedback.
        """
        (start, end) = (np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        delta = end - start
        squared = float(np.dot(delta, delta))
        if squared < 0.05 ** 2:
            return

        def contradicted(contact):
            return contact.blocks_segment(start, end, self.robot_radius_m)
        self._collision_memory = [c for c in self._collision_memory if not contradicted(c)]
        self._collision_patches = [(c, t) for (c, t) in self._collision_patches if not contradicted(c)]

    def _age_collision_patches(self) -> None:
        self._collision_patches = [(contact, ttl - 1) for (contact, ttl) in self._collision_patches if ttl > 1]

    def _raw_frontier_mask(self) -> np.ndarray:
        """Geometric visibility boundary, independent of navigation eligibility."""
        if self.occ.size == 0:
            return np.zeros(self.occ.shape, dtype=bool)
        (height, width) = self.occ.shape
        unknown = self._visibility_grid() == UNKNOWN
        frontier = np.zeros(self.occ.shape, dtype=bool)
        free = self._visibility_grid() == FREE
        for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(unknown)
            y0 = max(0, dy)
            y1 = height + min(0, dy)
            x0 = max(0, dx)
            x1 = width + min(0, dx)
            shifted[y0:y1, x0:x1] = unknown[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            frontier |= free & shifted
        edge = np.zeros(self.occ.shape, dtype=bool)
        if self.edge_frontier.shape == frontier.shape:
            edge = free & self._visibility_grid('edge_frontier') & self._unknown_beside(diagonal=True)
            frontier |= edge
        return frontier

    def frontier_clusters(self, *, include_pending: bool=False) -> list[dict]:
        v = self.visibility_map
        arrays = [self.occ, self.edge_frontier, self.frontier_seen]
        if v is not None:
            arrays.extend([v.occ, v.edge_frontier])
        key = (include_pending, tuple(((a.shape, hash(a.tobytes())) for a in arrays)), repr(self.origin), repr(None if v is None else v.origin), repr(getattr(self, '_frontier_robot_xy', None)), repr(self._collision_patches), repr(self._collision_memory), self.robot_radius_m)
        cache = getattr(self, '_frontier_cache', {})
        if key in cache:
            (clusters, stats) = cache[key]
            self.frontier_filter_stats = stats.copy()
            return clusters
        clusters = self._compute_frontier_clusters(include_pending=include_pending)
        if len(cache) >= 4:
            cache.clear()
        cache[key] = (clusters, getattr(self, 'frontier_filter_stats', {}).copy())
        self._frontier_cache = cache
        return clusters

    def _compute_frontier_clusters(self, *, include_pending: bool=False) -> list[dict]:
        """Boundary clusters; pending candidates defer safety checks to route planning."""
        frontier = self._raw_frontier_mask()
        if not frontier.size:
            return []
        (height, width) = frontier.shape
        raw_count = int(np.count_nonzero(frontier))
        robot_xy = getattr(self, '_frontier_robot_xy', None)
        supported = raw_count
        if robot_xy is not None and (not include_pending):
            (sx, sy) = self._index(robot_xy)
            inside = 0 <= sy < self.occ.shape[0] and 0 <= sx < self.occ.shape[1]
            if inside and self.occ[sy, sx] == FREE:
                free = (self.occ == FREE).astype(np.uint8)
                (_, labels) = cv2.connectedComponents(free, connectivity=4)
                frontier &= labels == labels[sy, sx]
                body = self._traversable_mask(self.robot_radius_m).astype(np.uint8)
                body[sy, sx] = 1
                (_, labels) = cv2.connectedComponents(body, connectivity=4)
                reachable = labels == labels[sy, sx]
                (distance, nearest) = cv2.distanceTransformWithLabels((~reachable).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                (candidates_y, candidates_x) = np.where(reachable)
                frontier &= distance * self.cell_m <= VIEW_STANDOFF_M
                for (y, x) in zip(*np.where(frontier)):
                    i = int(nearest[y, x]) - 1
                    observer = self._center(int(candidates_x[i]), int(candidates_y[i]))
                    if self._visibility_occluded(observer, self._center(x, y)):
                        frontier[y, x] = False
                supported = int(np.count_nonzero(frontier))
            else:
                frontier[:] = False
        self.frontier_filter_stats = {'raw_cells': raw_count, 'body_clear_cells': supported, 'reachable_cells': int(np.count_nonzero(frontier))}
        self._stamp_frontier(frontier)
        if self.frontier_seen.shape == frontier.shape:
            frontier &= ~self.frontier_seen
        seen = np.zeros(frontier.shape, dtype=bool)
        clusters = []
        (ys, xs) = np.nonzero(frontier)
        for (y, x) in zip(ys.tolist(), xs.tolist()):
            if seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                (cy, cx) = stack.pop()
                cells.append((cx, cy))
                for (ny, nx) in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1), (cy - 1, cx - 1), (cy - 1, cx + 1), (cy + 1, cx - 1), (cy + 1, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or (nx >= width):
                        continue
                    if seen[ny, nx] or not frontier[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            if len(cells) * self.cell_m < MIN_CLUSTER * 0.1 and (not include_pending):
                continue
            centers = np.array([self._center(ix, iy) for (ix, iy) in cells], dtype=np.float64)
            clusters.append({'size': len(cells), 'centroid': centers.mean(axis=0), 'cells': cells})
        clusters.sort(key=lambda item: item['size'], reverse=True)
        return clusters

    def choose_action(self, xy, heading) -> str | None:
        """Walk to a reachable frontier scored by directional visibility gain.

            ``None`` means every remaining frontier is unreachable or already observed.
            """
        self.decisions = []
        self._decision_step += 1
        self._retry_frontiers = [(p, t) for p, t in self._retry_frontiers if t > self._decision_step]
        self._failed_frontiers = [(p, t) for p, t in self._failed_frontiers if t > self._decision_step]
        self._failed_viewpoints = [(p, t) for p, t in self._failed_viewpoints if t > self._decision_step]
        self.plan_xy = []
        self.goal_xy = None
        self.stop_reason = ''
        self._age_collision_patches()
        if self.origin is None:
            return None
        origin_xy = np.asarray(xy, dtype=np.float64).reshape(2)
        origin_heading = _unit(heading)
        self._current_heading = origin_heading.copy()
        views = self._view_history
        if not views or np.linalg.norm(origin_xy - views[-1][0]) > 0.1 or float(np.dot(origin_heading, views[-1][1])) < 0.95:
            views.append((origin_xy.copy(), origin_heading.copy()))
        self._frontier_robot_xy = origin_xy.copy()
        covered = int((self._visibility_grid() != UNKNOWN).sum()) if self.visibility_map is not None else int(self.viewed.sum())
        self._coverage_history.append(covered)
        self._coverage_history = self._coverage_history[-8:]
        self._approach_standoff = 0.75 if len(self._coverage_history) >= 8 and (covered - self._coverage_history[0]) * self.cell_m ** 2 < 0.3 else VIEW_STANDOFF_M
        self._motion_history.append((origin_xy.copy(),covered))
        window=getattr(self,"_low_gain_window",[])
        window.append((origin_xy.copy(),covered));window=window[-40:]
        self._low_gain_window=window
        if len(window)==40:
            gain=covered-window[0][1]
            travel=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(window,window[1:]))
            if gain*self.cell_m**2<0.50 and travel>1.:
                if self._committed_goal is not None:
                    self._failed_viewpoints.append((self._committed_goal.copy(),self._decision_step+96))
                if self._sticky_centroid is not None:
                    self._failed_frontiers.append((self._sticky_centroid.copy(),self._decision_step+96))
                self._committed_goal=None;self._sticky_centroid=None
                self._low_gain_window=[]
                self.decisions.append({"event":"low_gain_retarget","window_steps":40,
                                       "new_cells":gain,"travel_m":float(travel)})
        if not self._near_frontier_mode and len(self._motion_history)>=24:
            recent=self._motion_history[-24:]
            gain=covered-recent[0][1]
            travel=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(recent,recent[1:]))
            loop=any(np.linalg.norm(origin_xy-p)<.6 and max(np.linalg.norm(q[0]-p) for q in self._motion_history[j:])>=1.
                     for j,(p,_) in enumerate(self._motion_history[:-12]) if j>=len(self._motion_history)-60)
            # Returning through known space may be necessary to reach a new
            # committed target. Do not interrupt that route just for low gain.
            goal_is_new = self._committed_goal is not None and min(
                np.linalg.norm(self._committed_goal-p) for p,_ in self._motion_history)>.75
            if gain*self.cell_m**2<0.75 and travel>=2. and loop and not goal_is_new:
                # Observing a boundary from a distance does not prove that its
                # unknown side has been explored. Retry only unapproached edges.
                ys,xs=np.nonzero(self.frontier_seen & (self.occ==FREE) & self._unknown_beside(diagonal=True))
                revived=[]
                for iy,ix in zip(ys,xs):
                    center=self._center(ix,iy)
                    if min(np.linalg.norm(center-p) for p,_ in self._motion_history)>1.:
                        self.frontier_seen[iy,ix]=False
                        self._frontier_born[iy,ix]=0
                        revived.append(center)
                if revived:
                    self._near_frontier_mode=True
                    self._committed_goal = None
                    self._sticky_centroid = None
                    self._last_goal = None
                    self.recovery_info={"step":self._decision_step,"gain_m2":gain*self.cell_m**2,"revived_cells":len(revived)}
                    self.decisions.append(dict(event="unvisited_frontier_recovery",**self.recovery_info))
        if self._near_frontier_mode:self._approach_standoff=0.

        if self._external_observation is not None:
            goal, centroid = self._external_observation
            clusters = self.frontier_clusters(include_pending=True)
            nearby = [c for c in clusters if np.linalg.norm(np.asarray(c['centroid']) - centroid) < 0.8]
            cluster = min(nearby, key=lambda c: np.linalg.norm(np.asarray(c['centroid']) - centroid)) if nearby else dict(centroid=centroid, cells=[])
            action = self._observe_external_frontier(origin_xy, origin_heading, cluster)
            if action not in ('retired', 'retry'):
                return action
            self._external_observation = None
            if action == 'retired':
                self._finish_viewpoint(goal)
                self.decisions.append(dict(event='viewpoint_observed', goal=goal.tolist()))
        for _pass in range(6):
            clusters = self.frontier_clusters(include_pending=True)
            if not clusters:
                return self._scan_for_frontier(origin_xy, origin_heading)
            followed = self._follow_committed(origin_xy, origin_heading)
            if followed is not None:
                return followed
            chosen = self._select_viewpoint(origin_xy, clusters, allow_recent=False)
            if chosen is None:
                chosen = self._select_viewpoint(origin_xy, clusters, allow_recent=True)
            if chosen is None:
                break
            viewpoint, cluster, path, already = chosen
            if already:
                self._sticky_centroid = np.asarray(cluster['centroid'], dtype=np.float64)
                action = self._observe_external_frontier(origin_xy, origin_heading, cluster)
                if action == 'retry':
                    continue
                if action == 'retired':
                    self._finish_viewpoint(viewpoint)
                    continue
                if action is not None:
                    return action
                continue
            action = 'navigate'
            self._committed_goal = np.asarray(viewpoint, dtype=np.float64)
            self._scan_turns = 0
            self.decisions.append({'event': 'select_viewpoint', 'goal': self._committed_goal.tolist(), 'frontier_size': int(cluster['size']), 'born': self._cluster_born(cluster)})
            points = [origin_xy] + [self._center(*cell) for cell in path]
            self._remember_plan(points, self._committed_goal)
            self._sticky_centroid = np.asarray(cluster['centroid'], dtype=np.float64)
            return action
        clusters = self.frontier_clusters(include_pending=True)
        if not clusters:
            return self._scan_for_frontier(origin_xy, origin_heading)
        return self._scan_for_frontier(origin_xy, origin_heading, pending=True)

    def choose_intent(self, xy, heading):
        """Keep frontier selection/observation here; delegate translation externally."""
        action = self.choose_action(xy, heading)
        return dict(kind='navigate' if action == 'navigate' else 'observe' if action else 'stop', action=None if action == 'navigate' else action, goal=None if self._committed_goal is None else self._committed_goal.copy(), look_at=None if self._sticky_centroid is None else self._sticky_centroid.copy())

    def navigation_arrived(self, xy):
        """Arrival schedules the selected view; it never retires unseen frontier cells."""
        if self._committed_goal is None:
            return
        goal = self._committed_goal.copy()
        centroid = self._sticky_centroid.copy() if self._sticky_centroid is not None else goal.copy()
        if np.linalg.norm(np.asarray(xy) - goal) > (0.25 if self._near_frontier_mode else 0.5):
            self._failed_viewpoints.append((goal.copy(), self._decision_step + 64))
        self._external_observation = (goal, centroid)
        self._committed_goal = None
        self.decisions.append(dict(event='navigation_arrived', goal=goal.tolist(), actual_xy=np.asarray(xy).tolist(), look_at=centroid.tolist()))

    def navigation_failed(self, reason):
        """Cool down the failed candidate and let the existing ranking choose another."""
        if self._committed_goal is not None:
            self._failed_viewpoints.append((self._committed_goal.copy(), self._decision_step + 64))
        if self._sticky_centroid is not None:
            self._failed_frontiers.append((self._sticky_centroid.copy(), self._decision_step + 64))
        self.decisions.append(dict(event='navigation_failed', reason=reason, goal=None if self._committed_goal is None else self._committed_goal.tolist()))
        self._committed_goal = None
        self._sticky_centroid = None
        self._external_observation = None
        self._scan_turns = 0

    def _observe_external_frontier(self, xy, heading, cluster):
        """A successful call to retirement is not proof of observation progress."""
        point = np.asarray(xy, dtype=float)
        anchor = self._external_observe_anchor
        if anchor is None or np.linalg.norm(point - anchor) > 0.15:
            self._external_observe_anchor = point.copy()
            self._external_observe_turns = 0
        before = int(self.frontier_seen.sum())
        action = self._face_or_retire(xy, heading, cluster)
        retired = int(self.frontier_seen.sum()) - before
        if not cluster['cells'] or retired > 0:
            self._external_observe_turns = 0
            self._scan_turns = 0
            return action
        if action != 'retired' and self._external_observe_turns < 12:
            self._external_observe_turns += 1
            self._scan_turns = 0
            return action
        centroid = np.asarray(cluster['centroid'], dtype=float)
        self._retry_frontiers.append((centroid.copy(), self._decision_step + 64))
        self._failed_viewpoints.append((point.copy(), self._decision_step + 24))
        self._committed_goal = self._sticky_centroid = None
        self._external_observe_turns = 0
        self._scan_turns = 0
        self.decisions.append(dict(event='observation_no_progress', centroid=centroid.tolist(), observer_xy=point.tolist(), retired_cells=retired))
        return 'retry'

    def _external_viewpoint(self, point, cluster, default, detail):
        """Use the executor's reachability, and relocate after a failed view."""
        retry = any((np.linalg.norm(np.asarray(cluster['centroid']) - p) < 0.8 for (p, _) in self._retry_frontiers))
        candidates = [np.asarray(default)]
        if retry:
            centers = [self._center(*cell) for cell in cluster['cells']]
            centers = [p for p in centers if np.linalg.norm(p - point) > 0.8]
            if not centers:
                (ys, xs) = np.nonzero(self.occ == FREE)
                free = self.origin + (np.column_stack((xs, ys)) + 0.5) * self.cell_m
                centroid = np.asarray(cluster['centroid'])
                distance = np.linalg.norm(free - point, axis=1)
                free = free[(distance > 0.8) & (distance <= 2.0) & (np.linalg.norm(free - centroid, axis=1) <= FRONTIER_OBSERVATION_M)]
                angles = np.arctan2(free[:, 1] - centroid[1], free[:, 0] - centroid[0])
                centers = list(free[np.argsort(angles)])
            else:
                centers.sort(key=lambda p: float(np.linalg.norm(p - point)))
            candidates = [centers[i] for i in np.unique(np.linspace(0, len(centers) - 1, min(8, len(centers)), dtype=int))] if centers else []
        detail['path_attempts'] = []
        for candidate in candidates:
            if any((np.linalg.norm(candidate - p) < 0.8 for (p, _) in self._failed_viewpoints)):
                continue
            (route, diagnostics) = self._external_route_planner(candidate)
            attempt = dict(viewpoint=candidate.tolist(), executor='habitat', **diagnostics)
            detail['path_attempts'].append(attempt)
            if not route:
                continue
            endpoint = np.asarray(route[-1], dtype=float)
            if np.linalg.norm(candidate - point) > 0.5 and np.linalg.norm(endpoint - point) <= 0.5 or any((np.linalg.norm(endpoint - p) < 0.8 for (p, _) in self._failed_viewpoints)):
                attempt['result'] = 'projection_without_observation_progress'
                continue
            length = sum((float(np.linalg.norm(np.asarray(b) - a)) for (a, b) in zip(route, route[1:])))
            attempt['result'] = 'accepted'
            detail['viewpoint'] = endpoint.tolist()
            return (endpoint, [self._index(p) for p in route], int(math.ceil(length / self.cell_m)))
        return None

    def _follow_committed(self, xy, heading) -> str | None:
        """Keep walking to the chosen viewpoint. A new frontier does not replace it."""
        if self._committed_goal is None:
            return None
        goal = np.asarray(self._committed_goal, dtype=np.float64).reshape(2)
        self._remember_plan([xy, goal], goal)
        self._scan_turns = 0
        return 'navigate'

    def _view_gain(self, item, point):
        """Estimate unknown visibility per action; never query scene geometry.

        Unknown look-through is capped, not assumed to be traversable. Nearby
        observation positions are discounted only for matching view directions.
        The constants and tie breakers match the frozen 18-scene experiment.
        """
        viewpoint = np.asarray(item[3])
        centroid = np.asarray(item[4]['centroid'])
        direction = _unit(centroid - viewpoint)
        yaw = math.atan2(direction[1], direction[0])
        layer = self.visibility_map or self
        unknown = set()
        half = math.radians(self.hfov_deg / 2)
        for angle in np.linspace(yaw - half, yaw + half, 25):
            ray = np.array([math.cos(angle), math.sin(angle)])
            first_unknown = None
            for distance in np.arange(0.1, 4.01, 0.1):
                cell = layer._index(viewpoint + distance * ray)
                value = layer._value(cell)
                if value == int(OCCUPIED):
                    break
                if value == int(UNKNOWN):
                    if first_unknown is None:
                        first_unknown = distance
                    if distance - first_unknown > 1.5:
                        break
                    unknown.add(cell)
        visits = int(sum((np.linalg.norm(viewpoint - position) < 0.75 and float(np.dot(heading, direction)) > math.cos(math.radians(35)) for (position, heading) in self._view_history)))
        repeat = min(1.0, max(0, visits - 1) / 4.0)
        heading = getattr(self, '_current_heading', np.array([1.0, 0.0]))
        turn = math.acos(float(np.clip(np.dot(heading, direction), -1, 1))) / self.turn_rad
        travel = 0.0 if item[6] else item[2] * self.cell_m / self.forward_m
        gain = len(unknown) * layer.cell_m ** 2
        score = gain * (1.0 - 0.75 * repeat) / (2.0 + travel + turn)
        return (score, dict(predicted_unknown_m2=gain, similar_view_samples=visits, repeat_penalty=repeat, predicted_turns=turn, predicted_forward_steps=travel, view_direction=direction.tolist(), unknown_gain_per_action=score))

    def _select_viewpoint(self, xy, clusters, *, allow_recent: bool):
        """Pick a look-point about 1.5 m in front of the frontier, toward the robot."""
        start = self._index(xy)
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        ranked = []
        audit = {'event': 'frontier_candidate_audit', 'decision_step': self._decision_step, 'allow_recent': allow_recent, 'robot_xy': point.tolist(), 'last_goal': None if self._last_goal is None else np.asarray(self._last_goal).tolist(), 'failed_frontiers': [{'xy': np.asarray(p).tolist(), 'expires': t} for p, t in self._failed_frontiers], 'failed_viewpoints': [{'xy': np.asarray(p).tolist(), 'expires': t} for p, t in self._failed_viewpoints], 'candidates': []}
        self.decisions.append(audit)
        for cluster in clusters:
            centroid = np.asarray(cluster['centroid'], dtype=np.float64)
            detail = {'id': len(audit['candidates']), 'size': int(cluster['size']), 'born': self._cluster_born(cluster), 'centroid': centroid.tolist(), 'bounds': [np.min(np.asarray(cluster['cells']), axis=0).tolist(), np.max(np.asarray(cluster['cells']), axis=0).tolist()], 'status': 'checking'}
            audit['candidates'].append(detail)
            if any((np.linalg.norm(centroid - p) < 0.8 for p, _ in self._failed_frontiers)):
                detail.update(status='rejected', reason='failed_frontier_within_0.8m')
                continue
            viewpoint = self._viewpoint_toward_robot(point, centroid)
            detail['viewpoint'] = viewpoint.tolist()
            selected = self._external_viewpoint(point, cluster, viewpoint, detail)
            if selected is None:
                detail.update(status='rejected', reason='no_executor_path_or_view_progress')
                continue
            viewpoint, native_path, native_length = selected
            already = float(np.linalg.norm(viewpoint - point)) <= (0.25 if self._near_frontier_mode else 0.5)
            path = native_path
            length = 0 if already else native_length
            detail.update(status='eligible', already=already, path_cells=length)
            ranked.append((-self._cluster_born(cluster), -int(cluster['size']), length, viewpoint, cluster, path, already))
        if not ranked:
            return None
        scores = {id(item): self._view_gain(item, point) for item in ranked}
        for item in ranked:
            detail = next((d for d in audit['candidates'] if d['status'] == 'eligible' and np.allclose(d['centroid'], item[4]['centroid']) and (d['size'] == item[4]['size'])))
            detail.update(scores[id(item)][1])
        ranked.sort(key=lambda item: (-scores[id(item)][0], item[1], item[2], item[0]))
        for rank, item in enumerate(ranked, 1):
            detail = next((d for d in audit['candidates'] if d['status'] == 'eligible' and np.allclose(d['centroid'], item[4]['centroid']) and (d['size'] == item[4]['size'])))
            detail['rank'] = rank
            if rank == 1:
                detail['selected'] = True
        _born, _size, _length, viewpoint, cluster, path, already = ranked[0]
        return (viewpoint, cluster, path, already)

    def _stamp_frontier(self, frontier: np.ndarray) -> None:
        """Give cells that just became frontier a shared birth stamp."""
        if self._frontier_born.shape != frontier.shape:
            return
        fresh = frontier & (self._frontier_born == 0)
        if not np.any(fresh):
            return
        self._frontier_clock += 1
        self._frontier_born[fresh] = self._frontier_clock

    def _cluster_born(self, cluster) -> int:
        if self._frontier_born.size == 0:
            return 0
        (height, width) = self._frontier_born.shape
        born = 0
        for (ix, iy) in cluster['cells']:
            if iy < 0 or ix < 0 or iy >= height or (ix >= width):
                continue
            born = max(born, int(self._frontier_born[iy, ix]))
        return born

    def _viewpoint_toward_robot(self, xy, centroid) -> np.ndarray:
        """A point VIEW_STANDOFF_M in front of the frontier, on the robot's side.

        The point stays on the line between the robot and the frontier. If that
        spot is not free, the nearest free cell is used.
        """
        centroid = np.asarray(centroid, dtype=np.float64).reshape(2)
        delta = np.asarray(xy, dtype=np.float64).reshape(2) - centroid
        distance = float(np.linalg.norm(delta))
        if distance < 0.0001:
            return centroid.copy()
        stand = min(self._approach_standoff, distance)
        geometric = centroid + delta / distance * stand
        return self._nearest_free(geometric)

    def _nearest_free(self, xy) -> np.ndarray:
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        cell = self._index(point)
        if self._value(cell) == int(FREE) or self.occ.size == 0:
            return point
        (height, width) = self.occ.shape
        limit = max(1, int(round(3.0 / self.cell_m)))
        queued = {cell}
        queue = [cell]
        head = 0
        while head < len(queue):
            (ix, iy) = queue[head]
            head += 1
            if self._value((ix, iy)) == int(FREE):
                return self._center(ix, iy)
            if max(abs(ix - cell[0]), abs(iy - cell[1])) >= limit:
                continue
            for (nx, ny) in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if (nx, ny) in queued or ny < 0 or nx < 0 or (ny >= height) or (nx >= width):
                    continue
                queued.add((nx, ny))
                queue.append((nx, ny))
        return point

    def _finish_viewpoint(self, goal) -> None:
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        self._last_goal = goal.copy()
        self._committed_goal = None
        self._sticky_centroid = None

    def _scan_for_frontier(self, xy, heading, *, pending=False) -> str | None:
        """The current view is used up. Turn once around before deciding it is finished."""
        turns = max(1, int(round(360.0 / math.degrees(self.turn_rad))))
        if self._scan_turns >= turns:
            cooling = any((t > self._decision_step for (_, t) in self._failed_frontiers + self._failed_viewpoints))
            if pending and cooling:
                self._scan_turns = 0
                self.decisions.append(dict(event='frontier_cooldown_scan'))
            else:
                self.stop_reason = '仍有未探索 frontier，但没有可执行的观察路径' if pending else '周围 frontier 都已看过'
                return None
        self._scan_turns += 1
        left = _rotate(_unit(heading), self.turn_rad)
        self._remember_heading_plan(xy, left)
        return 'turn_left'

    def _face_or_retire(self, xy, heading, cluster) -> str | None:
        """Turn toward a frontier already inside the view distance, then retire it."""
        centroid = np.asarray(cluster['centroid'], dtype=np.float64)
        delta = centroid - xy
        distance = float(np.linalg.norm(delta))
        if distance < 0.0001:
            self._retire_cells(xy, heading, cluster)
            return 'retired'
        direction = delta / distance
        cross = float(heading[0] * direction[1] - heading[1] * direction[0])
        dot = float(np.clip(np.dot(heading, direction), -1.0, 1.0))
        if abs(math.atan2(cross, dot)) > ALIGN_RAD:
            self._remember_plan([xy, centroid], centroid)
            return 'turn_left' if cross > 0.0 else 'turn_right'
        self._retire_cells(xy, heading, cluster)
        return 'retired'

    def _retire_cells(self, xy, heading, cluster) -> None:
        """Retire frontier cells the camera is actually pointing at.

        Cells outside the field of view stay targets. One glance must not erase
        the whole boundary.
        """
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        view = _unit(heading)
        limit = 0.6 if self._near_frontier_mode else FRONTIER_OBSERVATION_M
        half = math.radians(self.hfov_deg * 0.5)
        for (ix, iy) in cluster['cells']:
            if iy < 0 or ix < 0 or iy >= self.frontier_seen.shape[0] or (ix >= self.frontier_seen.shape[1]):
                continue
            center = self._center(ix, iy)
            delta = center - point
            distance = float(np.linalg.norm(delta))
            if distance > limit or distance < 0.0001:
                continue
            direction = delta / distance
            cross = float(view[0] * direction[1] - view[1] * direction[0])
            dot = float(np.clip(np.dot(view, delta / distance), -1.0, 1.0))
            if abs(math.atan2(cross, dot)) > half:
                continue
            if self._edge_hides_unseen_side(ix, iy, point, view):
                continue
            if self._visibility_occluded(point, center):
                continue
            self.frontier_seen[iy, ix] = True

    def _edge_hides_unseen_side(self, ix: int, iy: int, origin, heading) -> bool:
        """A field-of-view edge stays until its unknown side is inside the view."""
        if self.visibility_map is not None:
            v = self.visibility_map
            if v.origin is None:
                return False
            return v._edge_hides_unseen_side(*v._index(self._center(ix, iy)), origin, heading)
        if not self._is_edge_frontier(ix, iy):
            return False
        point = np.asarray(origin, dtype=np.float64).reshape(2)
        view = _unit(heading)
        half = math.radians(self.hfov_deg * 0.5)
        for neighbor in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1), (ix - 1, iy - 1), (ix - 1, iy + 1), (ix + 1, iy - 1), (ix + 1, iy + 1)):
            if self._value(neighbor) != int(UNKNOWN):
                continue
            center = self._center(*neighbor)
            delta = center - point
            distance = float(np.linalg.norm(delta))
            if distance < 0.0001:
                return True
            direction = delta / distance
            cross = float(view[0] * direction[1] - view[1] * direction[0])
            dot = float(np.clip(np.dot(view, delta / distance), -1.0, 1.0))
            if abs(math.atan2(cross, dot)) > half or self._visibility_occluded(point, center):
                return True
        return False

    def _is_edge_frontier(self, ix: int, iy: int) -> bool:
        """True when this free cell is still an unobserved field-of-view edge."""
        if self.edge_frontier.shape != self.occ.shape:
            return False
        if iy < 0 or ix < 0 or iy >= self.occ.shape[0] or (ix >= self.occ.shape[1]):
            return False
        return bool(self.edge_frontier[iy, ix] and self.occ[iy, ix] == FREE and self._touches_unknown(ix, iy, diagonal=True))

    def _remember_plan(self, points, goal) -> None:
        self.plan_xy = [np.asarray(point, dtype=np.float64).reshape(2) for point in points]
        self.goal_xy = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(2)

    def _remember_heading_plan(self, xy, heading) -> None:
        if heading is None:
            return
        goal = np.asarray(xy, dtype=np.float64).reshape(2) + _unit(heading) * LOOKAHEAD_M
        self._remember_plan([xy, goal], goal)

    def status(self) -> dict:
        clusters = self.frontier_clusters(include_pending=True)
        return {'covered': int((self._visibility_grid() != UNKNOWN).sum()) if self.visibility_map is not None else int(self.viewed.sum()) if self.viewed.size else 0, 'frontiers': len(clusters)}

    def _traversable_mask(self, clearance: float | None=None) -> np.ndarray:
        radius = self.plan_clearance_m if clearance is None else float(clearance)
        if not self.occ.size:
            return np.zeros(self.occ.shape, dtype=bool)
        reach = int(math.ceil(radius / self.cell_m))
        offsets = np.arange(-reach, reach + 1, dtype=np.float64) * self.cell_m
        kernel = (np.hypot(offsets[:, None], offsets[None, :]) <= radius + 1e-12).astype(np.uint8)
        occupied = cv2.dilate((self.occ == OCCUPIED).astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        mask = (self.occ == FREE) & ~occupied
        return mask

    def _mark_disk(self, xy, radius: float, kind: int) -> None:
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        margin = radius + 2.0 * self.cell_m
        self._prepare(point)
        self._prepare(point - margin)
        self._prepare(point + margin)
        reach = int(math.ceil(radius / self.cell_m)) + 1
        center = self._index(point)
        for iy in range(center[1] - reach, center[1] + reach + 1):
            for ix in range(center[0] - reach, center[0] + reach + 1):
                if float(np.linalg.norm(self._center(ix, iy) - point)) > radius:
                    continue
                self._paint(ix, iy, kind)

    def _commit_edge_frontier(self, fresh: set[tuple[int, int]]) -> None:
        """Keep field-of-view edge cells that are free before an obstacle.

        A later view drops the mark once the cell no longer touches unknown space.
        """
        if self.edge_frontier.shape != self.occ.shape:
            return
        (ys, xs) = np.nonzero(self.edge_frontier)
        for (iy, ix) in zip(ys.tolist(), xs.tolist()):
            if (ix, iy) in fresh:
                continue
            if not self._touches_unknown(ix, iy, diagonal=True):
                self.edge_frontier[iy, ix] = False
        (height, width) = self.edge_frontier.shape
        for (ix, iy) in fresh:
            if iy < 0 or ix < 0 or iy >= height or (ix >= width):
                continue
            if self.occ[iy, ix] != FREE:
                continue
            if not self._touches_unknown(ix, iy, diagonal=True):
                continue
            self.edge_frontier[iy, ix] = True

    def _unknown_beside(self, *, diagonal: bool) -> np.ndarray:
        unknown = self._visibility_grid() == UNKNOWN
        beside = np.zeros(unknown.shape, dtype=bool)
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        if diagonal:
            offsets = offsets + ((-1, -1), (-1, 1), (1, -1), (1, 1))
        (height, width) = unknown.shape
        for (dy, dx) in offsets:
            y0 = max(0, dy)
            y1 = height + min(0, dy)
            x0 = max(0, dx)
            x1 = width + min(0, dx)
            beside[y0:y1, x0:x1] |= unknown[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        return beside

    def _touches_unknown(self, ix: int, iy: int, *, diagonal: bool=False) -> bool:
        neighbors = [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]
        if diagonal:
            neighbors.extend(((ix - 1, iy - 1), (ix - 1, iy + 1), (ix + 1, iy - 1), (ix + 1, iy + 1)))
        return any((self._value(cell) == int(UNKNOWN) for cell in neighbors))

    def _mark_ray(self, start, end, *, occupied_end: bool) -> list[tuple[int, int]]:
        start = np.asarray(start, dtype=np.float64).reshape(2)
        end = np.asarray(end, dtype=np.float64).reshape(2)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-06:
            return []
        direction = delta / length
        step = self.cell_m * 0.45
        count = max(1, int(math.ceil(length / step)))
        points = [start + direction * min(length, index * step) for index in range(count + 1)]
        self._prepare(start)
        self._prepare(end)
        cells = []
        previous = None
        for point in points:
            cell = self._index(point)
            if cell == previous:
                continue
            previous = cell
            cells.append(cell)
        free_cells = []
        for (ix, iy) in cells[:-1]:
            if self._value((ix, iy)) == OCCUPIED:
                return free_cells
            self._paint(ix, iy, FREE)
            free_cells.append((ix, iy))
        (ix, iy) = cells[-1]
        if self._value((ix, iy)) == OCCUPIED:
            return free_cells
        if occupied_end:
            self._paint(ix, iy, OCCUPIED)
            return free_cells
        self._paint(ix, iy, FREE)
        free_cells.append((ix, iy))
        return free_cells

    def _paint(self, ix: int, iy: int, kind: int) -> None:
        current = self.occ[iy, ix]
        if kind == FREE and current == OCCUPIED:
            self.viewed[iy, ix] = True
            return
        self.occ[iy, ix] = kind
        self.viewed[iy, ix] = True

    def _prepare(self, xy) -> None:
        if self.origin is None:
            span = max(64, int(math.ceil(6.4 / self.cell_m)))
            self.origin = np.asarray(xy, dtype=np.float64).reshape(2) - span / 2.0 * self.cell_m
            self.occ = np.zeros((span, span), dtype=np.int8)
            self.viewed = np.zeros((span, span), dtype=bool)
            self.frontier_seen = np.zeros((span, span), dtype=bool)
            self.edge_frontier = np.zeros((span, span), dtype=bool)
            self._frontier_born = np.zeros((span, span), dtype=np.int32)
        (ix, iy) = self._index(xy)
        self._prepare_index(ix, iy)

    def _prepare_index(self, ix: int, iy: int) -> None:
        if self.origin is None:
            raise RuntimeError('grid origin is not set')
        pad_left = max(0, -ix) + (8 if ix < 2 else 0)
        pad_top = max(0, -iy) + (8 if iy < 2 else 0)
        (height, width) = self.occ.shape
        pad_right = max(0, ix - width + 1) + (8 if ix > width - 3 else 0)
        pad_bottom = max(0, iy - height + 1) + (8 if iy > height - 3 else 0)
        if pad_left or pad_right or pad_top or pad_bottom:
            self.occ = np.pad(self.occ, ((pad_top, pad_bottom), (pad_left, pad_right)))
            self.viewed = np.pad(self.viewed, ((pad_top, pad_bottom), (pad_left, pad_right)))
            self.frontier_seen = np.pad(self.frontier_seen, ((pad_top, pad_bottom), (pad_left, pad_right)))
            self.edge_frontier = np.pad(self.edge_frontier, ((pad_top, pad_bottom), (pad_left, pad_right)))
            self._frontier_born = np.pad(self._frontier_born, ((pad_top, pad_bottom), (pad_left, pad_right)))
            self.origin = self.origin - np.array([pad_left, pad_top], dtype=np.float64) * self.cell_m

    def _index(self, xy) -> tuple[int, int]:
        if self.origin is None:
            self._prepare(xy)
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        ix = int(math.floor((float(point[0]) - float(self.origin[0])) / self.cell_m))
        iy = int(math.floor((float(point[1]) - float(self.origin[1])) / self.cell_m))
        return (ix, iy)

    def _center(self, ix: int, iy: int) -> np.ndarray:
        origin = np.zeros(2) if self.origin is None else self.origin
        return origin + (np.array([ix, iy], dtype=np.float64) + 0.5) * self.cell_m

    def _value(self, cell: tuple[int, int]) -> int:
        (ix, iy) = cell
        if self.origin is None or iy < 0 or ix < 0 or (iy >= self.occ.shape[0]) or (ix >= self.occ.shape[1]):
            return int(UNKNOWN)
        return int(self.occ[iy, ix])

    def _occluded(self, origin_xy, target_xy) -> bool:
        """True when an occupied cell lies strictly between the camera and the target."""
        origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(target - origin)) < 1e-06:
            return False
        start = self._index(origin)
        end = self._index(target)
        for (ix, iy) in self._crossed_cells(origin, target):
            if (ix, iy) == start or (ix, iy) == end:
                continue
            if self._value((ix, iy)) == int(OCCUPIED):
                return True
        return False

    def _crossed_cells(self, start_xy, end_xy) -> list[tuple[int, int]]:
        """Every grid cell the segment enters, including both sides of a corner."""
        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        end = np.asarray(end_xy, dtype=np.float64).reshape(2)
        (ix, iy) = self._index(start)
        (end_ix, end_iy) = self._index(end)
        cells = [(ix, iy)]
        if (ix, iy) == (end_ix, end_iy):
            return cells
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        step_x = 0 if abs(dx) < 1e-12 else 1 if dx > 0.0 else -1
        step_y = 0 if abs(dy) < 1e-12 else 1 if dy > 0.0 else -1
        cell = self.cell_m
        origin_x = float(self.origin[0])
        origin_y = float(self.origin[1])
        if step_x == 0:
            t_max_x = math.inf
            t_delta_x = math.inf
        else:
            boundary_x = origin_x + (ix + (1 if step_x > 0 else 0)) * cell
            t_max_x = (boundary_x - float(start[0])) / dx
            t_delta_x = cell / abs(dx)
        if step_y == 0:
            t_max_y = math.inf
            t_delta_y = math.inf
        else:
            boundary_y = origin_y + (iy + (1 if step_y > 0 else 0)) * cell
            t_max_y = (boundary_y - float(start[1])) / dy
            t_delta_y = cell / abs(dy)
        limit = abs(end_ix - ix) + abs(end_iy - iy) + 2
        for _ in range(limit):
            if ix == end_ix and iy == end_iy:
                break
            if t_max_x + 1e-09 < t_max_y:
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y + 1e-09 < t_max_x:
                iy += step_y
                t_max_y += t_delta_y
            else:
                cells.append((ix + step_x, iy))
                cells.append((ix, iy + step_y))
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            cells.append((ix, iy))
        return cells
