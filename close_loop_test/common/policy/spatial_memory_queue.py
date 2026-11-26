"""Merge an incoming block into a short queue tail in shared odometry space.

Exporters already anchor simulator/LingBot/Map2 geometry to robby_world (Z-up).
Consequently both points and their source-camera poses use the same odometry
frame: their relative transform is identity, not another camera-pose transform.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np

def _write_metadata(path, metadata):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.json.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def short_tail(stats):
    """Strict thresholds, with 0.1 mm rounding for nominal 1 m odometry."""
    return round(float(stats['distance_m']), 4) < 1.0 or int(stats['frames']) < 10

def _source_stats(path, metadata, records):
    xyz = np.asarray(records['agent_xyz'], dtype=np.float64)
    if xyz.shape != (int(records['n']), 3) or not np.isfinite(xyz).all():
        raise ValueError('invalid memory odometry positions')
    odom = xyz[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    return dict(frames=int(metadata['total_frames_processed']), distance_m=float(np.linalg.norm(np.diff(odom[:, :2], axis=0), axis=1).sum()), first_odom_xyz=odom[0].tolist(), last_odom_xyz=odom[-1].tolist(), source_blocks=[dict(name=path.parent.name, render_manifests=metadata.get('render_manifests', []), render_manifest_fingerprint=metadata.get('render_manifest_fingerprint'), artifact_integrity=metadata.get('artifact_integrity', {}))])

def enqueue_memory(incoming, tail, records):
    """Return the active path; merge only when the *previous* tail is short.

    Keep the tail's identity so planner lists and EpisodeRecorder replace one
    entry. Concatenate corresponding rows without averaging ViT features or
    dropping viewpoints. Runtime NPZ replacement is atomic, preserving prior
    recorder hard links. The original frame queues/builders are untouched.
    """
    incoming = Path(incoming)
    meta_path = incoming.with_name('build_metadata.json')
    metadata = json.loads(meta_path.read_text())
    stats = _source_stats(incoming, metadata, records)
    metadata['queue_stats'] = stats
    if tail is None or Path(tail) == incoming or (not Path(tail).is_file()):
        _write_metadata(meta_path, metadata)
        return (incoming, None)
    tail = Path(tail)
    old_meta_path = tail.with_name('build_metadata.json')
    previous = json.loads(old_meta_path.read_text())
    previous_stats = previous.get('queue_stats')
    if previous_stats is None or not short_tail(previous_stats):
        _write_metadata(meta_path, metadata)
        return (incoming, None)
    if tail.parent.parent != incoming.parent.parent or previous.get('scene_id') != metadata.get('scene_id'):
        raise ValueError('cannot merge memory blocks across scenes')
    for value in (previous, metadata):
        if value.get('world_frame') != 'robby_world':
            raise ValueError('memory merge requires shared robby_world odometry coordinates')
        if value.get('frame_poses', {}).get('coordinate_system') != 'robby_world':
            raise ValueError('source camera poses must also be in robby_world')
    for key in ('backbone', 'feature_dim', 'voxel_size'):
        if previous.get(key) != metadata.get(key):
            raise ValueError(f'incompatible memory {key}')
    gap = float(np.linalg.norm(np.asarray(previous_stats['last_odom_xyz'])[:2] - np.asarray(stats['first_odom_xyz'])[:2]))
    combined_stats = dict(frames=previous_stats['frames'] + stats['frames'], distance_m=previous_stats['distance_m'] + gap + stats['distance_m'], first_odom_xyz=previous_stats['first_odom_xyz'], last_odom_xyz=stats['last_odom_xyz'], source_blocks=previous_stats['source_blocks'] + stats['source_blocks'])
    merged = dict(previous, queue_stats=combined_stats, total_frames_processed=combined_stats['frames'])
    detail = dict(incoming_block=incoming.parent.name, tail_block=tail.parent.name, previous_tail_frames=previous_stats['frames'], previous_tail_distance_m=previous_stats['distance_m'], frames=combined_stats['frames'], distance_m=combined_stats['distance_m'], operation='concatenate_in_shared_odometry_frame', tail_from_incoming=np.eye(4).tolist())
    merged['queue_merge'] = detail
    for key in ('render_manifest_fingerprint', 'feature_extractor_geometries', 'dedup_cache', 'point_filter', 'timing_s', 'build_time_s', 'best_obs_qual_voxels', 'artifact_integrity'):
        merged.pop(key, None)
    for key in ('frames_before_dedup', 'dedup_dropped_frames', 'frames_after_dedup', 'frames_after_stride', 'missing_rgbd_frames'):
        if key in previous and key in metadata:
            merged[key] = previous[key] + metadata[key]
    merged['render_manifests'] = previous.get('render_manifests', []) + metadata.get('render_manifests', [])
    with tempfile.NamedTemporaryFile(dir=tail.parent, suffix='.npz.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with np.load(tail, allow_pickle=False) as a, np.load(incoming, allow_pickle=False) as b:
            required = {'positions', 'features', 'rgb', 'counts', 'frame_poses'}
            if set(a.files) != required or set(b.files) != required:
                raise ValueError('unexpected spatial memory arrays; refusing to drop fields')
            (na, nb) = (len(a['positions']), len(b['positions']))
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for key in sorted(required):
                    (left, right) = (a[key], b[key])
                    if left.shape[0] != na or right.shape[0] != nb or left.shape[1:] != right.shape[1:] or (left.dtype != right.dtype):
                        raise ValueError(f'incompatible row-aligned memory array: {key}')
                    values = np.concatenate((left, right), axis=0)
                    if key == 'positions':
                        merged['positions_range'] = {axis: [float(values[:, i].min()), float(values[:, i].max())] for (i, axis) in enumerate('xyz')}
                    if key == 'counts':
                        merged['mean_count_per_voxel'] = float(values.mean())
                        merged['max_count_per_voxel'] = int(values.max())
                    with archive.open(key + '.npy', 'w', force_zip64=True) as out:
                        np.lib.format.write_array(out, values, allow_pickle=False)
                    del left, right, values
        merged['num_voxels'] = na + nb
        merged['frame_poses'] = dict(previous['frame_poses'], shape=[na + nb, 7])
        size = temporary.stat().st_size
        with temporary.open('rb') as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda : stream.read(1024 * 1024), b''):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        merged['file_size_mb'] = size / 1000000.0
        merged['artifact_integrity'] = {'spatial_mem.npz': dict(size_bytes=size, sha256=digest)}
        _write_metadata(old_meta_path, merged)
        os.replace(temporary, tail)
        _write_metadata(incoming.with_name('merged_source_metadata.json'), metadata)
        incoming.unlink()
        meta_path.unlink()
        return (tail, detail)
    finally:
        temporary.unlink(missing_ok=True)
