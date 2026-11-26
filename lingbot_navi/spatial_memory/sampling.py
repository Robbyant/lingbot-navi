"""Stable first-source-row voxel sampling shared by both adapters."""
import numpy as np

def first_voxel_indices(xyz, grid):
    xyz = np.asarray(xyz, dtype=np.float32)
    if not np.isfinite(grid) or grid <= 0:
        raise ValueError("sampling grid must be positive and finite")
    _, indices = np.unique(np.floor(xyz / grid).astype(np.int64), axis=0, return_index=True)
    return np.sort(indices)
