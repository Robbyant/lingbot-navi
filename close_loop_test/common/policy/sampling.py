"""Persist deterministic evaluation voxel representatives after the 4 cm build.

Coordinates are sampled as float32, exactly as load_spatial_tokens does. Keep
source order and every row-aligned array; never average features or move points.
The builder's voxel_size remains provenance, not the inference sampling grid.
"""
from lingbot_navi.spatial_memory.sampling import first_voxel_indices
from lingbot_navi.spatial_memory.io import write_evaluation_block
import json
from pathlib import Path
import numpy as np

def validate_grid(path, grid):
    metadata_path = Path(path).with_name('build_metadata.json')
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text())
    sampling = metadata.get('evaluation_sampling')
    if sampling and sampling['grid_size'] != grid:
        raise ValueError('stored evaluation memory grid differs from checkpoint grid')

def compact_memory(path, grid):
    """Atomically replace a built/merged NPZ, preserving existing hardlink snapshots.

    Deterministic first representatives obey D(D(A) + B) == D(A + B), so
    sampling a queue tail again after concatenation preserves inference inputs.
    """
    path = Path(path)
    if not np.isfinite(grid) or grid <= 0:
        raise ValueError('sampling grid must be positive and finite')
    validate_grid(path, grid)
    metadata_path = path.with_name('build_metadata.json')
    metadata = json.loads(metadata_path.read_text())
    required = {'positions', 'features', 'rgb', 'counts', 'frame_poses'}
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != required:
            raise ValueError('unexpected memory arrays; refusing to discard fields')
        xyz = np.asarray(source['positions'], dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or (not len(xyz)) or (not np.isfinite(xyz).all()):
            raise ValueError('invalid memory positions')
        selected = first_voxel_indices(xyz, grid)
        arrays = {}
        for key in source.files:
            values = source[key]
            if values.shape[0] != len(xyz):
                raise ValueError(f'memory array is not row aligned: {key}')
            arrays[key] = values[selected]
    metadata['evaluation_sampling'] = dict(grid_size=float(grid), random=False, method='first_source_row_per_voxel_float32', input_rows=len(xyz), output_rows=len(selected), source_build_voxel_size=metadata.get('voxel_size'))
    metadata['num_voxels'] = len(selected)
    if 'frame_poses' in metadata:
        metadata['frame_poses'] = dict(metadata['frame_poses'], shape=list(arrays['frame_poses'].shape))
    metadata['positions_range'] = {axis: [float(arrays['positions'][:, i].min()), float(arrays['positions'][:, i].max())] for (i, axis) in enumerate('xyz')}
    metadata['mean_count_per_voxel'] = float(arrays['counts'].mean())
    metadata['max_count_per_voxel'] = int(arrays['counts'].max())
    return write_evaluation_block(path, arrays, metadata)
