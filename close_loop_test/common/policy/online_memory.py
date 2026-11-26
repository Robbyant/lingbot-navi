from __future__ import annotations
from lingbot_navi.spatial_memory.observations import prepare_observations
from lingbot_navi.spatial_memory.builder import MemoryBuilder
from lingbot_navi.spatial_memory.sampling import first_voxel_indices
from lingbot_navi.spatial_memory.io import write_evaluation_block
import hashlib, io, math, zlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
def camera_pose_xyz_xyzw(camera_pose_c2w: np.ndarray) -> np.ndarray:
    """Convert a 4x4 camera transform to ``xyz + quaternion xyzw``."""
    pose = np.asarray(camera_pose_c2w, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f'camera_pose_c2w must be a finite 4x4 matrix, got {pose.shape}')
    rotation = pose[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(max(trace + 1.0, 0.0)) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('camera pose contains an invalid rotation')
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return np.concatenate((pose[:3, 3], quaternion)).astype(np.float32)
def canonical_frame(rgb, depth):
    """Keep offline JPEG95 and uint16-mm depth semantics, entirely in memory."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError('RGB/depth must be aligned HxWx3 / HxW arrays')
    (ok, encoded) = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    rgb_bytes = encoded.tobytes()
    valid = np.isfinite(depth) & (depth > 0.0)
    mm = np.zeros(depth.shape, dtype=np.uint16)
    mm[valid] = np.clip(np.rint(depth[valid] * 1000.0), 1.0, 65535.0).astype(np.uint16)
    (ok, encoded_depth) = cv2.imencode('.png', mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError('depth encoding failed')
    keys = (hashlib.sha256(rgb_bytes).digest(), hashlib.sha256(encoded_depth).digest())
    with Image.open(io.BytesIO(rgb_bytes)) as image:
        decoded = np.asarray(image.convert('RGB'))
    return (decoded, mm.astype(np.float32) / 1000.0, keys)
class OnlineMemoryBuilder:

    def __init__(self, extractor, *, scene, width, height, hfov, cam_height, max_voxels=500000):
        self.extractor = extractor
        self.scene = str(scene)
        (self.width, self.height) = (int(width), int(height))
        self.cam_height = float(cam_height)
        focal = self.width / 2.0 / math.tan(math.radians(float(hfov)) / 2.0)
        self.intrinsic = np.array([[focal, 0.0, self.width / 2.0], [0.0, focal, self.height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.max_voxels = max_voxels
        self.observations = []
        (self.seen_rgb, self.seen_depth) = (set(), set())
        self.frames = self.processed = 0

    def push_metric_frame(self, rgb, depth, intrinsic, pose):
        """Shared entry for simulator, Map and Map2 metric FLU geometry."""
        pose = np.asarray(pose, dtype=np.float32)
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError('expected FLU camera pose 4x4 and intrinsic 3x3')
        if not np.isfinite(pose).all() or not np.isfinite(intrinsic).all():
            raise ValueError('non-finite frame geometry')
        index = self.frames
        self.frames += 1
        (rgb, depth, keys) = canonical_frame(rgb, depth)
        if depth.shape != (self.height, self.width):
            raise ValueError('frame shape differs from segment camera')
        if keys[0] in self.seen_rgb or keys[1] in self.seen_depth:
            return False
        self.seen_rgb.add(keys[0])
        self.seen_depth.add(keys[1])
        extracted = self.extractor.extract(rgb)
        geometry = extracted.get('spatial_geometry') or {key: getattr(self.extractor, key) for key in ('input_height', 'input_width', 'patch_height', 'patch_width')}
        observations = prepare_observations(extracted['features'], torch.from_numpy(depth), torch.from_numpy(intrinsic), torch.from_numpy(pose), extracted.get('patch_rgb'), **geometry, min_depth=0.1, max_depth=10.0)
        self.observations.append((observations, f'frame_{index:06d}', torch.from_numpy(camera_pose_xyz_xyzw(pose))))
        self.processed += 1
        return True

    def finish(self, directory, grid, *, segment_name=None):
        """Select before saving; never write the unsampled 4cm archive."""
        directory = Path(directory)
        identity = self.scene + '/' + (segment_name or directory.name)
        self.seed = zlib.crc32(identity.encode('utf-8')) & 4294967295
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        accumulator = MemoryBuilder(voxel_size=0.04, feature_dim=int(self.extractor.feature_dim), max_voxels=self.max_voxels, preferred_min_depth=2.0, preferred_max_depth=5.0, generator=generator)
        for (observations, frame_name, pose) in self.observations:
            accumulator.add_observation(observations, frame_name=frame_name, frame_pose=pose)
        if accumulator.num_voxels == 0:
            raise ValueError('no valid memory observations')
        if grid is not None and (not np.isfinite(grid) or grid <= 0):
            raise ValueError('online memory requires a positive evaluation grid')
        memory = accumulator.finalize()
        self.observations.clear()
        xyz = np.asarray(memory.positions, dtype=np.float32)
        if grid is None:
            selected = np.arange(len(xyz))
        else:
            selected = first_voxel_indices(xyz, grid)
        arrays = {key: getattr(memory, key)[selected] for key in ('positions', 'features', 'rgb', 'counts', 'frame_poses')}
        metadata = dict(backbone='qwen_vit_merged_35_4b', feature_dim=self.extractor.feature_dim, scene_id=self.scene, world_frame='robby_world', voxel_size=0.04, point_select='best_obs', seed=0, scene_seed=self.seed, num_voxels=len(selected), frame_poses=dict(coordinate_system='robby_world', shape=list(arrays['frame_poses'].shape)), total_frames_processed=self.processed, frames_before_dedup=self.frames, frames_after_dedup=self.processed, frames_after_stride=self.processed, dedup_dropped_frames=self.frames - self.processed, missing_rgbd_frames=0, input_interface='online_rgbd_v1', evaluation_sampling=dict(grid_size=None if grid is None else float(grid), random=False, method='first_source_row_per_voxel_float32', input_rows=len(xyz), output_rows=len(selected), source_build_voxel_size=0.04), positions_range={axis: [float(arrays['positions'][:, i].min()), float(arrays['positions'][:, i].max())] for (i, axis) in enumerate('xyz')}, mean_count_per_voxel=float(arrays['counts'].mean()), max_count_per_voxel=int(arrays['counts'].max()))
        if grid is None:
            metadata.pop('evaluation_sampling')
        return write_evaluation_block(directory / 'spatial_mem.npz', arrays, metadata,
                                      invalidate_metadata=True)
