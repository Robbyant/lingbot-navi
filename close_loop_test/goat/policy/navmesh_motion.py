"""Numerically stable no-sliding steps; no obstacle bypass or route search."""
import json
import numpy as np
INSET_M = 1e-05
TOL_M = 5e-07
START_HORIZONTAL_TOL_M = 0.2
START_VERTICAL_TOL_M = 0.2

def compatible_navmesh_start(pathfinder, start, *, max_vertical_m=START_VERTICAL_TOL_M):
    """Validate a nearby mesh reference without moving the original body pose."""
    start = np.asarray(start, dtype=np.float64)
    snapped = np.asarray(pathfinder.snap_point(start), dtype=np.float64)
    delta = snapped - start
    horizontal = float(np.linalg.norm(delta[[0, 2]]))
    vertical = float(abs(delta[1]))
    valid = np.isfinite(start).all() and np.isfinite(snapped).all() and (horizontal <= START_HORIZONTAL_TOL_M + TOL_M) and (vertical <= max_vertical_m + TOL_M) and pathfinder.is_navigable(snapped)
    detail = dict(valid=bool(valid), original_position=start.tolist(), mesh_reference=snapped.tolist(), horizontal_gap_m=horizontal, vertical_gap_m=vertical, max_horizontal_m=START_HORIZONTAL_TOL_M, max_vertical_m=max_vertical_m, position_changed=False)
    if not valid:
        raise ValueError('invalid_navmesh_start: ' + json.dumps(detail))
    detail['status'] = 'on_navmesh' if pathfinder.is_navigable(start) else 'boundary_compatible'
    return (snapped, detail)


def stable_no_slide(pathfinder, start, end, report=None):
    """Keep boundary release and nearby off-mesh entry on the command ray.

    An inward seed must lie on the mesh and connect back to the original
    point under the same no-sliding filter. Never use try_step (sliding),
    snap the robot, search sideways, or extend the requested displacement.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    requested = end - start
    distance = float(np.linalg.norm(requested))
    # Habitat also filters rotations; don't let those perturb the position.
    if distance < 1e-10:
        return start.copy()
    off_mesh = not pathfinder.is_navigable(start)
    if off_mesh:
        try:
            compatible_navmesh_start(pathfinder, start)
        except ValueError:
            return start.copy()
    raw = np.asarray(pathfinder.try_step_no_sliding(start, end), dtype=np.float64)
    result = raw.copy()
    event = None
    if off_mesh:
        # Detour's no-sliding query can itself snap an outside start sideways or
        # backwards. Accept only a real step along the requested horizontal ray.
        direction = requested[[0, 2]]
        norm2 = float(direction @ direction)
        fraction = float((raw-start)[[0, 2]] @ direction / norm2) if norm2 else -1.
        lateral = (raw-start)[[0, 2]]-fraction*direction
        if (not np.isfinite(raw).all() or not 0 < fraction <= 1+1e-6
                or np.linalg.norm(lateral) > 2*TOL_M
                or abs(raw[1]-start[1]) > START_VERTICAL_TOL_M+TOL_M
                or not pathfinder.is_navigable(raw)):
            return start.copy()
        event = 'offmesh_entry'

    def on_mesh(point):
        snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64)
        return (np.isfinite(snapped).all() and pathfinder.is_navigable(point)
                and np.linalg.norm(snapped - point) <= TOL_M)

    if (distance > 2 * INSET_M and np.linalg.norm(raw - start) <= TOL_M
            and on_mesh(start)):
        # Only the commanded direction is eligible; outward seeds fail on_mesh.
        seed = (start + requested / distance * INSET_M).astype(np.float32)
        if on_mesh(seed):
            reverse = np.asarray(pathfinder.try_step_no_sliding(seed, start))
            if np.linalg.norm(reverse - start) <= TOL_M:
                trial = np.asarray(pathfinder.try_step_no_sliding(seed, end), dtype=np.float64)
                # Keep horizontal motion on the command ray, even if the mesh
                # adjusts height on a slope. This forbids tangential sliding.
                direction = requested[[0, 2]]
                norm2 = float(direction @ direction)
                fraction = float((trial - start)[[0, 2]] @ direction / norm2) if norm2 else -1
                lateral = (trial - start)[[0, 2]] - fraction * direction
                if (np.isfinite(trial).all() and 0 < fraction <= 1 + 1e-6
                        and np.linalg.norm(lateral) <= 2 * TOL_M
                        and np.linalg.norm(trial - start) > 2 * INSET_M
                        and on_mesh(trial)):
                    result = trial
                    event = "boundary_release"
    # Stop just inside the traversed segment after a partial collision.
    moved = float(np.linalg.norm(result - start))
    if moved > 2 * INSET_M and np.linalg.norm(result - end) > INSET_M:
        inset = (result - (result - start) / moved * INSET_M).astype(np.float32)
        checked = np.asarray(pathfinder.try_step_no_sliding(start, inset))
        if on_mesh(inset) and np.linalg.norm(checked - inset) <= TOL_M:
            result = np.asarray(inset, dtype=np.float64)
            event = event or "collision_inset"
    if event and report is not None:
        report(dict(event=event, start=start.tolist(), requested=end.tolist(),
                    raw=raw.tolist(), result=result.tolist(), inset_m=INSET_M,
                    allow_sliding=False))
    return result



def install_boundary_filter(sim):
    """Install only on the robot; leave simulator config and goal camera alone."""
    import magnum as mn
    if sim.config.sim_cfg.allow_sliding:
        return
    original = sim.agents[0].controls.move_filter_fn

    def step(start, end):
        if not sim.pathfinder.is_loaded:
            return original(start, end)
        report = None if getattr(sim, "_navmesh_follower_planning", False) else (
            lambda data: print("[navmesh-boundary] " + json.dumps(data), flush=True))
        result = stable_no_slide(sim.pathfinder, start, end, report=report)
        return mn.Vector3(result)

    sim.agents[0].controls.move_filter_fn = step
