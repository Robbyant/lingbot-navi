"""Shared lossless RGB-D recording and JSON serialization."""
from pathlib import Path
import numpy as np
from .geometry import camera_intrinsics

def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_observation(env, observations, output, *, map1=False):
    from .memory_camera import simulator_camera
    state = env.sim.get_agent(0).get_state()
    camera = state.sensor_states['rgb']
    q = camera.rotation
    specs = env.sim.get_agent(0).agent_config.sensor_specifications
    height = float(next(spec.position[1] for spec in specs if spec.uuid == 'rgb'))
    pose = simulator_camera(state.position, [q.x, q.y, q.z, q.w], height)
    rgb = np.asarray(observations['rgb'])[..., :3]
    depth = np.asarray(observations['depth'], dtype=np.float32).reshape(480, 640)
    intrinsic = camera_intrinsics(rgb.shape[1], rgb.shape[0], 90.)
    extra = {}
    if map1:
        from .memory_camera import habitat_agent_pose_to_zup_camera, _quat_to_matrix_xyzw
        xyz, rotation = habitat_agent_pose_to_zup_camera(state.position, [q.x,q.y,q.z,q.w], height)
        reference = np.eye(4)
        reference[:3,:3] = _quat_to_matrix_xyzw(*rotation)
        reference[:3,3] = xyz
        extra['reference_camera'] = reference
    np.savez(output / 'observation.npz', rgb=rgb, depth=depth, intrinsic=intrinsic, camera_pose=pose, agent_position=state.position, **extra)
    return depth, intrinsic, pose

