"""Greedy coverage-preserving subsets of exported robby-world memories."""
from collections import Counter
from pathlib import Path
import numpy as np
COVERAGE_CELL_M = 0.25

def memory_coverage(paths):
    """Occupied XY cells using all finite points in the shared Z-up frame.

    Exports already use robby_world (identity in worker._materialize).
    This measures observed cloud footprint, not inferred free space.
    """
    result = {}
    for raw in paths:
        path = str(raw)
        with np.load(Path(path), allow_pickle=False) as archive:
            key = 'positions' if 'positions' in archive.files else 'xyz'
            xyz = np.asarray(archive[key], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f'invalid memory point cloud: {path}')
        xy = xyz[np.isfinite(xyz).all(axis=1), :2]
        cells = np.unique(np.floor(xy / COVERAGE_CELL_M).astype(np.int64), axis=0)
        result[path] = set(map(tuple, cells.tolist()))
    return result
