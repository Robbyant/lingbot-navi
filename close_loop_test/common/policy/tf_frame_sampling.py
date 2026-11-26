"""Whole-exploration calibration sampling using level-depth quality and coverage."""
import numpy as np
from .localization_tf import heading, wrap

DEPTH_NEAR_M = 0.5
DEPTH_FAR_M = 15.0
MIN_VALID_FRACTION = 0.6
MIN_USEFUL_FRACTION = 0.5


def depth_quality(depth):
    """Central 60% of synchronized level metric depth; never persist the image."""
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2 or not d.size:
        raise ValueError('level calibration depth must be nonempty HxW')
    h, w = d.shape
    d = d[int(h*.2):max(int(h*.8), int(h*.2)+1),
          int(w*.2):max(int(w*.8), int(w*.2)+1)]
    # At most ~20k samples at normal resolution, without another render.
    stride = max(1, min(d.shape)//120)
    d = d[::stride, ::stride].astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    values = d[valid]
    vf = float(valid.mean())
    near = float(np.mean(values < DEPTH_NEAR_M)) if values.size else 0.
    far = float(np.mean(values > DEPTH_FAR_M)) if values.size else 0.
    useful = float(np.mean((values >= DEPTH_NEAR_M) & (values <= DEPTH_FAR_M))) if values.size else 0.
    quantiles = np.quantile(values, [.1, .5, .9]).tolist() if values.size else [None]*3
    median = quantiles[1]
    reasons = []
    if vf < MIN_VALID_FRACTION: reasons.append('insufficient_valid_depth')
    if useful < MIN_USEFUL_FRACTION: reasons.append('insufficient_useful_depth')
    if median is not None and median < DEPTH_NEAR_M: reasons.append('too_near')
    if median is not None and median > DEPTH_FAR_M: reasons.append('too_far')
    proximity = min(1., median/DEPTH_NEAR_M, DEPTH_FAR_M/median) if median else 0.
    return dict(accepted=not reasons, score=float(.65*vf*useful+.2*vf+.15*proximity),
                reasons=reasons, valid_fraction=vf, useful_fraction=useful,
                near_fraction=near, far_fraction=far,
                p10_m=quantiles[0], median_m=median, p90_m=quantiles[2],
                near_m=DEPTH_NEAR_M, far_m=DEPTH_FAR_M, roi='central_60_percent')


def select_frames(candidates, qualities, limit=24):
    """Good frames first, then descending quality bands for compensation.

    Within each tier cover least-represented time strata first, then maximize
    XY/heading novelty and depth quality. Distinct frames may share a station
    when needed to fill the budget; fitting still performs its own consensus.
    """
    candidates = sorted(candidates, key=lambda s: s[0])
    if not candidates:
        return [], dict(quality_passed=0, quality_rejected=0, compensation_count=0,
                        compensation_steps=[], temporal_bins_covered=0)
    poses = np.asarray([s[2] for s in candidates], dtype=float)
    xy = poses[:, :2]
    yaw = np.asarray([heading(p) for p in poses])
    steps = np.asarray([s[0] for s in candidates])
    bins = np.minimum(limit-1, ((steps-steps.min())*limit//max(1, int(steps.max()-steps.min()+1)))).astype(int)
    q = [qualities.get(s[0], {}) for s in candidates]
    passed = np.asarray([bool(v.get('accepted', False)) for v in q])
    quality = np.asarray([np.clip(float(v.get('score', 0.)), 0., 1.) for v in q])
    tiers = np.where(passed, 11, np.floor(quality*10).astype(int))
    selected = []
    occupancy = np.zeros(limit, dtype=int)
    for tier in sorted(set(tiers), reverse=True):
        available = [i for i in range(len(candidates)) if tiers[i] == tier]
        while available and len(selected) < limit:
            if selected:
                delta = xy[available, None, :] - xy[selected][None, :, :]
                angles = wrap(yaw[available, None] - yaw[selected][None, :])
                novelty = np.min(np.sum(delta**2, axis=2)/4 + (angles/np.deg2rad(60))**2, axis=1)
                novelty = novelty/(1+novelty)
            else:
                novelty = np.zeros(len(available))
            scores = {i:float(novelty[k]+quality[i]) for k,i in enumerate(available)}
            index = max(available, key=lambda i:(-occupancy[bins[i]], scores[i], -int(steps[i])))
            selected.append(index);occupancy[bins[index]] += 1;available.remove(index)
        if len(selected) == limit: break
    selected.sort(key=lambda i: steps[i])
    fallback = [int(steps[i]) for i in selected if not passed[i]]
    return [candidates[i] for i in selected], dict(
        quality_passed=int(passed.sum()), quality_rejected=int((~passed).sum()),
        compensation_count=len(fallback), compensation_steps=fallback,
        temporal_bins_covered=int(np.count_nonzero(occupancy)),
        candidate_step_range=[int(steps.min()), int(steps.max())],
        selected_quality=[dict(step=int(steps[i]), **q[i]) for i in selected])


def extend_ladder_frames(selected,history,target):
 available=[x for x in history if x['step'] not in {s['step'] for s in selected}]
 lo,hi=min(x['step'] for x in history),max(x['step'] for x in history)
 def binid(x):return min(23,(x['step']-lo)*24//max(1,hi-lo+1))
 while available and len(selected)<target:
  poses=np.asarray([x['odom_camera'] for x in selected]);yaw=np.asarray([heading(x) for x in poses])
  occupied=np.bincount([binid(x) for x in selected],minlength=24)
  def score(x):
   q=x.get('depth_quality',{});p=np.array(x['odom_camera']);d=np.sum((poses[:,:2]-p[:2])**2,axis=1)/4+(wrap(yaw-heading(p))/np.deg2rad(60))**2
   novelty=float(d.min());return (bool(q.get('accepted')), -int(occupied[binid(x)]),novelty/(1+novelty)+float(q.get('score',0)),-x['step'])
  x=max(available,key=score);selected.append(x);available.remove(x)
 return selected
