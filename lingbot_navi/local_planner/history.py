"""Causal pose keyframes shared by offline datasets and online evaluation.

Thresholds compare the current pose with the last accepted keyframe, not with
the previous observation. Translation is Euclidean displacement (meters), and
rotation is the shortest quaternion angle (degrees). No future pose is needed.
"""
from collections import deque
import math
import numpy as np


class PoseKeyframes:
    def __init__(self, config):
        self.config = config
        self.indices = deque(maxlen=config.history_frames + 1)
        self.last_pose = None
        self.last_index = -1
        self.accepted = False
        self.translation_m = self.rotation_deg = 0.0

    def observe(self, index, position, rotation_xyzw):
        if type(index) is not int or index != self.last_index + 1:
            raise ValueError("keyframe observations must be sequential from index zero")
        p, q = np.asarray(position, dtype=float), np.asarray(rotation_xyzw, dtype=float)
        if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
                or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-3)):
            raise ValueError("keyframe pose requires finite XYZ and unit XYZW quaternion")
        q = q / np.linalg.norm(q)
        distance = angle = 0.0
        if self.last_pose is not None:
            old_p, old_q = self.last_pose
            distance = float(np.linalg.norm(p - old_p))
            angle = math.degrees(2 * math.acos(float(np.clip(abs(q @ old_q), 0, 1))))
        self.accepted = (self.last_pose is None or
                         distance + 1e-7 >= self.config.history_distance_m or
                         angle + 1e-6 >= self.config.history_rotation_deg)
        if self.accepted:
            self.indices.append(index)
            self.last_pose = p.copy(), q.copy()
        self.last_index = index
        self.translation_m, self.rotation_deg = distance, angle
        return self.accepted

    def select(self, reference, current):
        if type(reference) is not int or current != self.last_index or not 0 <= reference <= current:
            raise ValueError("keyframe reference/current must be causal observed indices")
        history = [i for i in self.indices if i < current]
        history = history[-self.config.history_frames:] if self.config.history_frames else []
        indices = sorted(set(history + [reference, current]))
        roles = np.asarray([[i in history, i == reference, i == current] for i in indices], np.float32)
        return indices, roles

    def diagnostics(self):
        return dict(policy="pose_keyframes_v1", accepted_keyframe=self.accepted,
                    keyframe_indices=list(self.indices), translation_since_keyframe_m=self.translation_m,
                    rotation_since_keyframe_deg=self.rotation_deg,
                    distance_threshold_m=self.config.history_distance_m,
                    rotation_threshold_deg=self.config.history_rotation_deg)
