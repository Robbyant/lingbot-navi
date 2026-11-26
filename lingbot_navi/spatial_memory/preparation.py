from __future__ import annotations
from pathlib import Path
import numpy as np
import hashlib
from .sampling import first_voxel_indices

def unit_quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("expected finite xyzw quaternion")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1, atol=1e-3, rtol=0):
        raise ValueError("expected unit xyzw quaternion")
    return q / norm

def quaternion_product(a, b):
    a, b = unit_quaternion(a), unit_quaternion(b)
    return np.r_[
        a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
        a[3] * b[3] - np.dot(a[:3], b[:3]),
    ]

def rotation_matrix(q):
    x, y, z, w = unit_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

def _rotation_quaternion(seed, scene_id, block_id, degrees):
    values = [
        int.from_bytes(
            hashlib.blake2b(
                f"{seed}\0{scene_id}\0{block_id}\0block_rotation_{axis}".encode(),
                digest_size=8,
            ).digest(),
            "big",
        )
        / 2**64
        for axis in ("roll", "pitch", "yaw")
    ]
    roll, pitch, yaw = (2 * np.asarray(values) - 1) * np.deg2rad(degrees)
    qx = np.array([np.sin(roll / 2), 0., 0., np.cos(roll / 2)])
    qy = np.array([0., np.sin(pitch / 2), 0., np.cos(pitch / 2)])
    qz = np.array([0., 0., np.sin(yaw / 2), np.cos(yaw / 2)])
    return quaternion_product(qz, quaternion_product(qy, qx))


def _rotation(seed, scene_id, block_id, degrees):
    return rotation_matrix(_rotation_quaternion(seed, scene_id, block_id, degrees))


def _arrays(path, feature_dim):
    with np.load(path, allow_pickle=False) as data:
        key = "xyz" if "xyz" in data else "positions"
        xyz = np.asarray(data[key], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32)
        if (
            "xyz" in data
            and "positions" in data
            and not np.array_equal(data["xyz"], data["positions"])
        ):
            raise ValueError("Conflicting xyz/positions")
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not len(xyz)
        or features.shape != (len(xyz), feature_dim)
    ):
        raise ValueError(
            f"{path}: expected nonempty XYZ [N,3] and features [N,{feature_dim}]"
        )
    if not np.isfinite(xyz).all() or not np.isfinite(features).all():
        raise ValueError(f"{path}: memory contains nonfinite values")
    return xyz, features


def prepare_memory(block_paths, block_ids, scene_key, *, preparation, feature_dim=2560):
    """Prepare ordered blocks once and return the transforms actually applied."""
    preparation.validate()
    if not block_paths or len(block_paths) != len(block_ids):
        raise ValueError("Memory needs matching nonempty block paths and IDs")
    if any(not isinstance(name, str) or not name for name in block_ids):
        raise ValueError("Every block needs a nonempty block_id")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("Every block needs a unique block_id")
    all_xyz, all_features, offsets, transforms = [], [], [0], []
    for path, name in zip(block_paths, block_ids):
        xyz, features = _arrays(path, feature_dim)
        if preparation.voxel_size:
            indices = first_voxel_indices(xyz, preparation.voxel_size)
            xyz, features = xyz[indices], features[indices]
        rotation = _rotation(
            preparation.seed, scene_key, name, preparation.rotation_degrees
        )
        center = xyz.mean(axis=0, dtype=np.float64)
        xyz = ((xyz.astype(np.float64) - center) @ rotation.T).astype(np.float32)
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        all_xyz.append(np.ascontiguousarray(xyz[order]))
        all_features.append(np.ascontiguousarray(features[order]))
        offsets.append(offsets[-1] + len(xyz))
        prepared_from_local = np.eye(4)
        prepared_from_local[:3, :3] = rotation
        prepared_from_local[:3, 3] = -rotation @ center
        transforms.append(prepared_from_local)
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Memory exceeds int32 block-offset capacity")
    return dict(
        memory_xyz=np.concatenate(all_xyz),
        memory_features=np.concatenate(all_features),
        block_cu_seqlens=np.asarray(offsets, dtype=np.int32),
    ), transforms
