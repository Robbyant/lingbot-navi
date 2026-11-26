"""Forward-motion scale and first-camera anchoring shared with Qwen lingbot-map1.0."""
import numpy as np

FORWARD_THRESHOLD_M = .001


def align_forward(predicted, reference, threshold=FORWARD_THRESHOLD_M):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    if predicted.shape != reference.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise ValueError('Map1/reference pose shapes must match [N,4,4]')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError('Non-finite Map1/reference poses')
    gt = np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1)
    pr = np.linalg.norm(np.diff(predicted[:, :3, 3], axis=0), axis=1)
    moving = gt > threshold
    denominator = pr[moving].sum()
    if not moving.any() or denominator <= 1e-8:
        raise ValueError('Map1 metric scale is unobservable: no usable forward translation')
    scale = float(gt[moving].sum() / denominator)
    rotation = reference[0, :3, :3] @ predicted[0, :3, :3].T
    translation = reference[0, :3, 3] - scale * rotation @ predicted[0, :3, 3]
    aligned = predicted.copy()
    aligned[:, :3, :3] = rotation @ predicted[:, :3, :3]
    aligned[:, :3, 3] = scale * predicted[:, :3, 3] @ rotation.T + translation
    return aligned, dict(path_scale=scale,forward_segments=int(moving.sum()),excluded_segments=int((~moving).sum()),forward_threshold_m=threshold,pose_alignment='first_camera_anchor_forward_length',rotation=rotation.tolist(),translation=translation.tolist())
