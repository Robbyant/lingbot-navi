"""lingbot-map1.0 geometry feeding the existing release feature/memory builder."""
from collections import deque
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .client import Map1MemoryClient, map1_builder_rgb
from .stream import GeometryStream, HISTORY_FRAMES, MAX_GEOMETRY_FRAMES
from ..policy.online_memory import OnlineMemoryBuilder
from ..policy.spatial_memory_queue import short_tail

OPENCV_FROM_FLU = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


class Map1Memory:
    def __init__(self, owner, settings, device):
        self.owner = owner
        self.client = Map1MemoryClient(device=device, **settings)
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.stream = None
        self.frames = []
        self.positions = []

    def observe(self, bundle, frame_id):
        # Deliberately do not read simulator depth or the simulator FLU pose.
        with np.load(bundle, allow_pickle=False) as data:
            rgb = np.array(data['rgb'], copy=True)
            reference = np.array(data['reference_camera'], copy=True)
            position = np.array(data['agent_position'], copy=True)
        if self.stream is None:
            self.stream = GeometryStream(self.client, str(frame_id), history=self.history)
        try:
            online = self.stream.push(rgb, reference, frame_id)
        except BaseException:
            self.close()
            raise
        self.frames.append(rgb)
        self.positions.append(position)
        return dict(frames=len(self.frames), backend='lingbot_map1',
                    scale_valid=online['scale_valid'], reason=online.get('reason'))

    def finish(self, segment, phase):
        stream, self.stream = self.stream, None
        frames, self.frames = self.frames, []
        positions, self.positions = self.positions, []
        result = dict(state='skipped', source_phase=phase, paths=list(self.owner.paths),
                      backend='lingbot_map1', skip_reason='insufficient_frames')
        if stream is None:
            return result
        try:
            blocks, tail = stream.finish(len(frames))
        finally:
            stream.abort()
        result.update(tail_frames_skipped=tail, scale_skipped=stream.skipped_blocks)
        if not blocks:
            if stream.skipped_blocks:
                result['skip_reason'] = 'scale_unobservable'
            return result
        root = self.owner.output/'memories'
        root.mkdir(parents=True, exist_ok=True)
        published = []
        # Build every part before publishing any; never merge independently
        # estimated learned blocks into a preceding simulator/learned memory.
        with tempfile.TemporaryDirectory(prefix='.map1-', dir=root) as temporary:
            staged = []
            try:
                for index, (start, end, prediction) in enumerate(blocks, 1):
                    xyz = np.asarray(positions[start:end])
                    distance = float(np.linalg.norm(np.diff(xyz[:, [0, 2]], axis=0), axis=1).sum())
                    if short_tail(dict(frames=end-start, distance_m=distance)):
                        result['skip_reason'] = 'short_learned_memory'
                        continue
                    name = (f'{segment}_part{index:04d}of{len(blocks):04d}'
                            if len(frames) > MAX_GEOMETRY_FRAMES else segment)
                    directory = Path(temporary)/name
                    target = root/name
                    if target.exists():
                        raise FileExistsError(f'Memory segment already exists: {target}')
                    rgb0 = map1_builder_rgb(frames[start])
                    builder = OnlineMemoryBuilder(self.owner.extractor, scene=self.owner.scene,
                        width=rgb0.shape[1], height=rgb0.shape[0], hfov=90, cam_height=1.5,
                        max_voxels=self.owner.max_voxels)
                    basis = np.eye(4); basis[:3, :3] = OPENCV_FROM_FLU
                    poses = np.einsum('nij,jk->nik', prediction['aligned_poses_c2w'], basis)
                    scale = prediction['alignment']['path_scale']
                    for i, rgb in enumerate(frames[start:end]):
                        depth = prediction['depth'][i]
                        if depth.ndim == 3: depth = depth[..., 0]
                        builder.push_metric_frame(map1_builder_rgb(rgb), depth*scale,
                                                  prediction['intrinsics'][i], poses[i])
                    metadata = builder.finish(directory, self.owner.preparation.voxel_size)
                    metadata['geometry_segment'] = dict(index=index, count=len(blocks),
                        frame_range=[start, end], source_phase=phase, pose_source='lingbot_map1',
                        depth_source='lingbot_map1', geometry_metadata=prediction['metadata'],
                        **prediction['alignment'])
                    (directory/'build_metadata.json').write_text(json.dumps(metadata, indent=2))
                    staged.append((directory, target))
                for directory, target in staged:
                    directory.rename(target)
                    published.append(target)
            except BaseException:
                for target in published: shutil.rmtree(target)
                raise
        paths = [str(directory/'spatial_mem.npz') for directory in published]
        if paths:
            self.owner.paths.extend(paths)
            self.owner.cache_key = None
            result.update(state='done', path=paths[0], paths=list(self.owner.paths),
                          published_paths=paths, split_batch=len(frames)>MAX_GEOMETRY_FRAMES)
            result.pop('skip_reason', None)
        return result

    def close(self):
        # Closing stdin ends the private worker even if the parent was
        # interrupted mid-request; do not send another protocol request here.
        self.client.close()
        self.stream = None
        self.frames.clear()
        self.positions.clear()
        self.history.clear()
