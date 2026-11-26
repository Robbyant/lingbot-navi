"""Task-local coverage selection from model-predicted block poses."""
from pathlib import Path
import numpy as np


def rigid_frame(frame):
    if abs(float(frame.get('scale', 1.0)) - 1.0) > 1e-05:
        raise ValueError('task memory selection requires metric rigid poses')
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(frame['rotation_matrix'], dtype=float)
    matrix[:3, 3] = np.asarray(frame['translation'], dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=0.0001) or (not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=0.0001)):
        raise ValueError('invalid block rigid pose')
    return matrix

def select_coverage(memories, local_xyz, predictions, newest_first, *, horizontal_rotation=None, cell_m=0.25, min_gain=0.02):
    """Retain a block iff its unique cells add >2 percentage points of total area.

    Geometry is assembled exclusively with predicted inter-block poses. The
    optional rotation only removes the first block's local augmentation so XY
    measures horizontal coverage, rather than a tilted projection.
    """
    if not len(memories) == len(local_xyz) == len(predictions) or not memories:
        raise ValueError('missing or incomplete all-memory predictions')
    if not np.allclose(rigid_frame(predictions[0]), np.eye(4), atol=1e-05):
        raise ValueError('model predictions must use the first block as identity anchor')
    if sorted(newest_first) != list(range(len(memories))):
        raise ValueError('recency order must contain every block exactly once')
    rotation = np.eye(3) if horizontal_rotation is None else np.asarray(horizontal_rotation)
    cells = []
    for (xyz, prediction) in zip(local_xyz, predictions):
        tf = rigid_frame(prediction)
        points = np.asarray(xyz, dtype=float).reshape(-1, 3)
        points = (points @ tf[:3, :3].T + tf[:3, 3]) @ rotation.T
        if not np.isfinite(points).all():
            raise ValueError('non-finite coverage geometry')
        grid = np.floor(points[:, :2] / cell_m).astype(np.int64)
        cells.append(set(map(tuple, grid.tolist())))
    total = set().union(*cells)
    if not total:
        raise ValueError('all-memory coverage is empty')
    (covered, chosen, decisions) = (set(), [], [])
    for i in newest_first:
        added = len(cells[i] - covered)
        accepted = added > min_gain * len(total)
        decisions.append(dict(path=str(memories[i]), area_m2=len(cells[i]) * cell_m ** 2, added_area_m2=added * cell_m ** 2, gain=added / len(total), accepted=accepted))
        if accepted:
            chosen.append(i)
            covered.update(cells[i])
    if not chosen:
        raise ValueError(f'no block adds more than {min_gain:.0%} coverage; cannot select a TF anchor')
    return dict(selected=[str(memories[i]) for i in chosen], selected_indices=chosen, total_area_m2=len(total) * cell_m ** 2, selected_area_m2=len(covered) * cell_m ** 2, coverage=len(covered) / len(total), cell_m=cell_m, min_gain=min_gain, decisions=decisions)

def anchor_bridge(old_target_frame, full_old_frame, predicted_new_frame, full_new_frame, selected_new_frame):
    """new-input-local <- old-TF-local, using the predicted cross-block relation.

    Reference matrices only bridge *the same physical block* before/after
    recentering or index-dependent augmentation. They never replace the predicted
    transform between the old and new target blocks.
    """
    return np.linalg.inv(rigid_frame(selected_new_frame)) @ rigid_frame(full_new_frame) @ np.linalg.inv(rigid_frame(predicted_new_frame)) @ np.linalg.inv(rigid_frame(full_old_frame)) @ np.asarray(old_target_frame)

def memory_versions(memories):
    return {str(p): (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in memories}

def append_task_memories(selected, previous_versions, pool):
    """Append new/rewritten blocks, including a previously excluded merged tail."""
    versions = memory_versions(pool)
    missing = set(selected) - set(versions)
    if missing:
        raise ValueError(f'locked task memory disappeared: {sorted(missing)}')
    added = [str(p) for p in pool if str(p) not in selected and previous_versions.get(str(p)) != versions[str(p)]]
    return ([*selected, *added], versions, added)

def record_exploration_gain(state, added_m2):
    """Drop one oldest pre-anchor block after two consecutive low-gain rounds.

    Only task-start blocks after the anchor are eligible. Rewritten/new blocks
    are protected by the caller, and deleted blocks stay out for this task even after rewriting.
    If no block can be removed, retain the streak so the caller can close exploration
    and request one final retrieval. A successful removal starts a fresh two-round window.
    """
    state['low_gain_streak'] = state.get('low_gain_streak', 0) + 1 if added_m2 < 1.0 else 0
    if state['low_gain_streak'] < 2:
        return None
    eligible = state.setdefault('prunable', list(reversed(state['selected'][1:])))
    while eligible:
        oldest = eligible.pop(0)
        if oldest in state['selected'] and oldest != state['selected'][0]:
            state['low_gain_streak'] = 0
            state['selected'].remove(oldest)
            state.setdefault('removed', set()).add(oldest)
            return oldest
    return None

def memory_frame_counts(memories):
    """Read observation counts, including all sources of a merged block."""
    import json
    counts = []
    for path in memories:
        metadata = json.loads((Path(path).parent / 'build_metadata.json').read_text())
        count = metadata.get('queue_stats', {}).get('frames')
        if count is None:
            count = metadata.get('total_frames_processed')
        if count is None or int(count) < 0:
            raise ValueError(f'missing/invalid frame count: {path}')
        counts.append(int(count))
    return counts

def select_frame_anchor(selection, frame_counts):
    """Promote the newest selected block with >30 frames; otherwise most frames.

    Preserve recency separately: moving the anchor to input index zero must not
    make newer blocks eligible for old-memory pruning.
    """
    order = list(selection['selected_indices'])
    paths = dict(zip(order, selection['selected']))
    anchor = next((i for i in order if frame_counts[i] > 30), None)
    fallback = anchor is None
    if fallback:
        anchor = max(order, key=lambda i: frame_counts[i])
    older = order[order.index(anchor) + 1:]
    selected = [anchor, *(i for i in order if i != anchor)]
    return dict(selection, selected=[paths[i] for i in selected], selected_indices=selected, selected_recency_order=[paths[i] for i in order], selected_frame_counts={paths[i]: frame_counts[i] for i in order}, anchor_frame_count=frame_counts[anchor], anchor_min_frames_exclusive=30, anchor_fallback_most_frames=fallback, prunable_oldest_first=[paths[i] for i in reversed(older)])
