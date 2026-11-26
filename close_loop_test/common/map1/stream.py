"""Episode-local warm starts and online metric depth, with sequence-end publication."""
from collections import deque
import hashlib
import numpy as np
from .geometry import align_forward

MAX_GEOMETRY_FRAMES = 250

HISTORY_FRAMES = 20


class GeometryStream:
    def __init__(self, client, token, *, history=None, cam_height=1.5):
        self.client = client
        self.token = token
        self.history = history if history is not None else deque(maxlen=HISTORY_FRAMES)
        self.cam_height = cam_height
        self.count = 0
        self.block_count = 0
        self.prefix_count = 0
        self.references = []
        self.online_poses = []
        self.blocks = []
        self.skipped_blocks = []
        self.scale_estimated = False
        self.closed = False
        self.active = False

    def _send_frame(self, rgb, reference):
        self.references.append(reference)
        reply = self.client.request(dict(op='frame', rgb=np.asarray(rgb, dtype=np.uint8), return_online=True))
        online = reply.get('online')
        if online is None:
            return dict(scale_valid=False, reason='model_warmup', depth=None)
        poses = np.asarray(online['poses_c2w'])
        if poses.ndim != 3 or poses.shape[1:] != (4,4) or not np.isfinite(poses).all():
            raise ValueError('Invalid online camera predictions')
        self.online_poses.extend(poses)
        if len(self.online_poses) != len(self.references):
            raise ValueError('Online pose/reference correspondence mismatch')
        if any(pose is None for pose in self.references):
            return dict(scale_valid=False, reason='reference_pose_unavailable', depth=None)
        try:
            aligned, summary = align_forward(self.online_poses, self.references)
        except ValueError as error:
            if 'unobservable' not in str(error):
                raise
            # Provisional online depth is explicitly uncalibrated, not a
            # successful metric estimate. Continue estimating on every arrival.
            aligned = np.asarray(self.online_poses).copy()
            rotation = self.references[0][:3,:3] @ aligned[0,:3,:3].T
            translation = self.references[0][:3,3] - rotation @ aligned[0,:3,3]
            aligned[:,:3,:3] = rotation @ aligned[:,:3,:3]
            aligned[:,:3,3] = aligned[:,:3,3] @ rotation.T + translation
            summary = dict(path_scale=1.0)
            scale_valid = False
        else:
            scale_valid = True
            self.scale_estimated = True
        depth = np.asarray(online['depth'], dtype=np.float32) * summary['path_scale']
        intrinsic = np.asarray(online['intrinsics'])
        if not np.isfinite(depth).all() or not np.isfinite(intrinsic).all():
            raise ValueError('Non-finite online metric geometry')
        return dict(scale_valid=scale_valid, scale_fallback=not scale_valid,
                    reason=None if scale_valid else "scale_unobservable",
                    scale_estimated=self.scale_estimated, depth=depth, pose_c2w=aligned[-1],
                    intrinsic=intrinsic, path_scale=summary['path_scale'],
                    scale_summary=summary, geometry_stage='online', prefix_frames=self.prefix_count)

    def push(self, rgb, reference_pose=None, frame_id=None):
        if self.closed:
            raise RuntimeError('Geometry stream is closed')
        if reference_pose is not None:
            reference_pose = np.array(reference_pose, dtype=np.float64, copy=True)
            if reference_pose.shape != (4,4) or not np.isfinite(reference_pose).all():
                raise ValueError('Invalid metric reference camera pose')
        if self.block_count == 0:
            self.client.begin_stream()
            self.active = True
            self.references, self.online_poses = [], []
            self.scale_estimated = False
            # A seq restart can reuse the previous seq's last observation. Replay
            # it only once, as the current frame, rather than twice in the prefix.
            prefix = [entry for entry in self.history
                      if frame_id is None or entry[0] != frame_id]
            self.prefix_count = len(prefix)
            for _, image, pose in prefix:
                self._send_frame(image, pose)
        result = self._send_frame(rgb, reference_pose)
        self.count += 1
        self.block_count += 1
        result.update(frame_id=frame_id, sequence_frame=self.count-1, token=self.token,
                      rgb_sha256=hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest())
        if reference_pose is not None:
            if frame_id is not None and self.history and self.history[-1][0] == frame_id:
                self.history.pop()
            self.history.append((frame_id, np.array(rgb, dtype=np.uint8, copy=True), reference_pose))
        if self.block_count == MAX_GEOMETRY_FRAMES:
            self._finish_block()
        return result

    def _finish_block(self):
        prediction = self.client.request(dict(op='predict'))
        total = self.prefix_count + self.block_count
        if any(len(prediction[k]) != total for k in ('poses_c2w','intrinsics')):
            raise ValueError('Geometry stream lost frame correspondence')
        # One depth per message stays below the IPC 64 MiB limit. Prefix pixels
        # never enter memory twice and need not be transferred back to the parent.

        if self.references and all(pose is not None for pose in self.references):
            try:
                aligned, summary = align_forward(prediction['poses_c2w'], self.references)
            except ValueError as error:
                if 'unobservable' not in str(error):
                    raise
            else:
                prediction['aligned_poses_c2w'] = aligned[self.prefix_count:]
                prediction['alignment'] = summary
                self.scale_estimated = True
        if 'alignment' not in prediction:
            self.skipped_blocks.append(dict(frame_range=[self.count-self.block_count,self.count],
                scale_estimated=self.scale_estimated,
                reason='final_scale_unobservable' if self.scale_estimated else 'scale_never_estimated'))
            self.client.request(dict(op='abort'))
            self.active = False
            self.block_count = 0
            self.references, self.online_poses = [], []
            return
        prediction['depth'] = np.stack([self.client.request(dict(op='depth', index=i))['depth']
                                       for i in range(self.prefix_count,total)])
        for key in ('poses_c2w','intrinsics'):
            prediction[key] = prediction[key][self.prefix_count:]
        prediction['metadata'] = dict(prediction.get('metadata', {}),
            history_window=HISTORY_FRAMES, prefix_frames=self.prefix_count,
            new_frames=self.block_count, model_input_frames=total, scale_estimated=True)
        self.blocks.append((self.count-self.block_count, self.count, prediction))
        self.client.request(dict(op='abort'))
        self.active = False
        self.block_count = 0
        self.references, self.online_poses = [], []

    def finish(self, expected_count):
        if self.closed or expected_count != self.count:
            raise ValueError('Geometry stream/frame log correspondence mismatch')
        skipped = self.block_count if self.block_count < 10 else 0
        if self.block_count >= 10:
            self._finish_block()
        elif self.active:
            self.client.request(dict(op='abort'))
            self.active = False
            self.block_count = 0
        self.closed = True
        blocks, self.blocks = self.blocks, []
        return blocks, skipped

    def abort(self):
        self.blocks.clear()
        self.references, self.online_poses = [], []
        self.closed = True
        if self.active:
            self.client.request(dict(op='abort'))
            self.active = False
            self.block_count = 0
