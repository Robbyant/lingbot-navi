"""Episode-local, gravity-aligned transform from odometry to the first memory."""
import numpy as np
from scipy.spatial.transform import Rotation

def camera_in_odom(position, quaternion):
    world = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    body = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    rotation = Rotation.from_matrix(world @ Rotation.from_quat(quaternion).as_matrix() @ body)
    return np.r_[world @ position, rotation.as_quat()]

def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def yaw_rotation(yaw):
    return Rotation.from_rotvec([0.0, 0.0, yaw])

def heading(pose):
    forward = Rotation.from_quat(pose[3:]).as_matrix()[:, 0]
    return np.arctan2(forward[1], forward[0])

class StaticLocalizationTF:
    noise = np.diag(np.square([0.6, 0.6, 0.35, np.deg2rad(6.0)]))
    gate = 13.28
    block_switch_noise_fraction = 0.1

    def __init__(self):
        self.pivot = None
        self.state = None
        self.cov = None
        self.pending = []
        self.candidate_limit = 24
        self.output_from_filter = np.eye(4)
        self.target_block = None
        self.target_frame = None

    def rebase(self, new_from_old, *, target_block, target_frame):
        """Change output coordinates, preserving the estimate and observation history.

        A different established target block adds uncertainty to the filter;
        initial attachment and same-block coordinate changes do not.
        The filter remains in its calibrated gravity-aligned coordinates. New
        localization observations are mapped back before updates, so arbitrary
        block augmentation rotations do not invalidate the yaw-only estimator.
        """
        transform = np.asarray(new_from_old, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or (not np.allclose(transform[3], [0, 0, 0, 1])) or (not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=0.0001)) or (not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=0.0001)):
            raise ValueError('invalid localization target transform')
        changed_block = self.target_block is not None and self.target_block != str(target_block)
        self.output_from_filter = transform @ self.output_from_filter
        if changed_block and self.cov is not None:
            self.cov = self.cov + self.block_switch_noise_fraction * self.noise
        self.target_block = str(target_block)
        self.target_frame = np.asarray(target_frame, dtype=float).copy()

    def _filter_observation(self, observation):
        if observation is None or np.array_equal(self.output_from_filter, np.eye(4)):
            return observation
        pose = np.asarray(observation, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-06:
            return observation
        inverse = np.linalg.inv(self.output_from_filter)
        xyz = inverse[:3, :3] @ pose[:3] + inverse[:3, 3]
        quat = (Rotation.from_matrix(inverse[:3, :3]) * Rotation.from_quat(pose[3:])).as_quat()
        return np.r_[xyz, quat]

    def add_candidate(self, odom, observation):
        return self._add_candidate(odom, self._filter_observation(observation))

    def update(self, odom, observation):
        return self._update(odom, self._filter_observation(observation))

    def _add_candidate(self, odom, observation):
        """Store a level, finite camera pair; truth never enters this buffer."""
        if abs(Rotation.from_quat(odom[3:]).as_matrix()[2, 0]) > np.sin(np.deg2rad(2.0)):
            return 'tilted'
        if observation is None:
            return 'no_localization'
        pose = np.asarray(observation, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-06:
            return 'invalid'
        self.pending.append((odom.copy(), pose.copy()))
        self.pending = self.pending[-self.candidate_limit:]
        return 'candidate'

    def fit_candidates(self, *, method='median', position_tolerance_m=1.25):
        """Fit yaw and metric translation using distinct camera stations."""
        if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
            raise ValueError('position tolerance must be finite and positive')
        if method not in {'median', 'huber'}:
            raise ValueError('unknown TF initialization method')
        if len(self.pending) < 3:
            return False
        (odom, block) = map(np.asarray, zip(*self.pending))
        stations = []
        for (i, position) in enumerate(odom[:, :2]):
            if all((np.linalg.norm(position - odom[j, :2]) >= 0.5 for j in stations)):
                stations.append(i)
        if len(stations) < 3:
            return False
        (odom, block) = (odom[stations], block[stations])
        yaws = np.array([wrap(heading(b) - heading(o)) for (o, b) in zip(odom, block)])
        pivot = np.median(odom[:, :3], axis=0)
        offsets = odom[:, :3] - pivot
        best = None
        for (i, yaw) in enumerate(yaws):
            centers = block[:, :3] - yaw_rotation(yaw).apply(offsets)
            distance = np.linalg.norm(centers - centers[i], axis=1)
            angles = abs(wrap(yaws - yaw))
            mask = (distance <= position_tolerance_m) & (angles <= np.deg2rad(15))
            count = int(mask.sum())
            if count < 3 or count * 2 <= len(stations):
                continue
            xy = odom[mask, :2]
            if np.max(np.linalg.norm(xy[:, None] - xy, axis=2)) < 1.5:
                continue
            score = (count, -float(np.median(distance[mask] + angles[mask])))
            if best is None or score > best[0]:
                best = (score, mask, yaw)
        if best is None:
            return False
        (_, mask, seed) = best
        yaw = wrap(seed + np.median(wrap(yaws[mask] - seed)))
        centers = block[mask, :3] - yaw_rotation(yaw).apply(offsets[mask])
        state = np.r_[np.median(centers, axis=0), yaw]
        fit = dict(stations=len(stations), inliers=int(mask.sum()), position_tolerance_m=float(position_tolerance_m))
        if method == 'huber':
            from scipy.optimize import least_squares
            sigma = np.sqrt(np.diag(self.noise))

            def residual(value):
                xyz = yaw_rotation(value[3]).apply(offsets[mask]) + value[:3] - block[mask, :3]
                angles = wrap(value[3] - yaws[mask])
                return np.c_[xyz / sigma[:3], angles / sigma[3]].ravel()
            try:
                solved = least_squares(residual, state, loss='huber', f_scale=1.345, ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1000)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                return False
            if not solved.success or not np.isfinite(solved.x).all() or (not np.isfinite(solved.cost)):
                return False
            state = solved.x.copy()
            state[3] = wrap(state[3])
            fit.update(method='huber', cost=float(solved.cost), nfev=int(solved.nfev))
        self.pivot = pivot
        self.state = state
        self.cov = self.noise.copy()
        self.last_fit = fit
        self.pending.clear()
        return True

    def _update(self, odom, observation):
        forward = Rotation.from_quat(odom[3:]).as_matrix()[:, 0]
        if abs(forward[2]) > np.sin(np.deg2rad(2.0)):
            return 'tilted'
        if observation is None:
            return 'no_localization'
        pose = np.asarray(observation, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-06:
            return 'invalid'
        if self.state is None:
            self._add_candidate(odom, pose)
            return 'initialized' if self.fit_candidates() else 'awaiting_confirmation'
        yaw = wrap(heading(pose) - heading(odom))
        offset = odom[:3] - self.pivot
        arm = yaw_rotation(self.state[3]).apply(offset)
        residual = np.r_[pose[:3] - self.state[:3] - arm, wrap(yaw - self.state[3])]
        H = np.eye(4)
        H[:3, 3] = [-arm[1], arm[0], 0.0]
        S = H @ self.cov @ H.T + self.noise
        if residual @ np.linalg.solve(S, residual) > self.gate:
            self._add_candidate(odom, pose)
            return 'recovered' if self.fit_candidates() else 'outlier'
        self.pending.clear()
        gain = np.linalg.solve(S, H @ self.cov).T
        self.state += gain @ residual
        self.state[3] = wrap(self.state[3])
        rest = np.eye(4) - gain @ H
        self.cov = rest @ self.cov @ rest.T + gain @ self.noise @ gain.T
        return 'updated'

    def matrix(self):
        if self.state is None:
            return None
        tf = np.eye(4)
        tf[:3, :3] = yaw_rotation(self.state[3]).as_matrix()
        tf[:3, 3] = self.state[:3] - tf[:3, :3] @ self.pivot
        return self.output_from_filter @ tf

    def predict(self, odom):
        tf = self.matrix()
        if tf is None:
            return None
        xyz = tf[:3, :3] @ odom[:3] + tf[:3, 3]
        quat = (Rotation.from_matrix(tf[:3, :3]) * Rotation.from_quat(odom[3:])).as_quat()
        return np.r_[xyz, quat]
