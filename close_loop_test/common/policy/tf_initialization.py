"""Episode observation selection and isolated, transactional subtask TF fitting."""
import numpy as np
from scipy.spatial.transform import Rotation
from .localization_tf import StaticLocalizationTF, heading, wrap

HISTORY_LIMIT = 128
INITIALIZATION_LIMIT = 24
TF_RETRY_LADDER = ((24, 2.0), (36, 2.5), (48, 3.0), (72, 4.0))


def should_capture(samples, step, odom):
    if not samples:
        return True
    previous_step, _, previous = samples[-1]
    return step != previous_step and (
        np.linalg.norm(odom[:2] - previous[:2]) >= 1.0
        or abs(wrap(heading(odom) - heading(previous))) >= np.deg2rad(25.))


def _level_station_samples(samples, limit=HISTORY_LIMIT):
    """Valid cached stations, newest first, without consulting localization."""
    for sample in reversed(samples if limit is None else samples[-limit:]):
        pose = np.asarray(sample[2], dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:]) < 1e-6):
            continue
        if abs(Rotation.from_quat(pose[3:]).as_matrix()[2, 0]) > np.sin(np.deg2rad(2.)):
            continue
        yield sample




def diverse_station_samples(samples):
    """Greedy XY/heading coverage, matching the validated offline selector.

    Keep the newest valid station, then maximize minimum joint distance to the
    selected set: (XY distance / 2 m)^2 + (wrapped yaw difference / 60 deg)^2.
    One vote per >=0.5 m station prevents turning in place from dominating the
    fit. Ties prefer newer observations; no localization or truth is consulted.
    """
    candidates = list(_level_station_samples(samples))
    if not candidates:
        return []
    poses = np.asarray([sample[2] for sample in candidates], dtype=float)
    xy = poses[:, :2]
    headings = np.asarray([heading(pose) for pose in poses])
    selected = []
    available = list(range(len(candidates)))
    while available and len(selected) < INITIALIZATION_LIMIT:
        if not selected:
            index = available[0]
        else:
            delta = xy[available, None, :] - xy[selected][None, :, :]
            angles = wrap(headings[available, None] - headings[selected][None, :])
            scores = np.min(np.sum(delta ** 2, axis=2) / 4
                            + (angles / np.deg2rad(60)) ** 2, axis=1)
            index = available[int(np.argmax(scores))]
        selected.append(index)
        available = [i for i in available
                     if all(np.linalg.norm(xy[i] - xy[j]) >= .5 for j in selected)]
    return [candidates[i] for i in selected]


def _initialize_tf_round(samples, memories, localize, *, observe=None, expected_frame=None, strict_errors=False, qualities=None, position_tolerance_m=1.25, selected_samples=None):
    """Use the full ordered subset; return a new TF only after a successful fit.

    Localize all selected diverse stations (up to 24) before consensus and Huber
    fitting. Reference frame metadata only labels/validates the output axes;
    it never enters the optimizer.
    Exceptions and failed fits leave the caller's previous TF untouched.
    """
    sampling = {}
    if selected_samples is not None:
        selected = selected_samples
    elif qualities is None:  # Historical/offline callers retain their original selector.
        selected = diverse_station_samples(samples)
    else:
        from .tf_frame_sampling import select_frames
        selected, sampling = select_frames(list(_level_station_samples(samples, limit=None)), qualities,
                                           INITIALIZATION_LIMIT)
    details = dict(method="huber", selection="spatial_heading_diversity",
                   sample_limit=INITIALIZATION_LIMIT, sample_steps=[s[0] for s in selected],
                   available_samples=len(samples), stations=len(selected),
                   position_tolerance_m=float(position_tolerance_m),
                   attempted=0, localized=0, errors=0, ready=False, reason="insufficient_stations")
    if qualities is not None:
        details.update(selection="whole_exploration_depth_temporal_diversity_v1", **sampling)
    if len(selected) < 3:
        return None, details
    candidate = StaticLocalizationTF()
    candidate.candidate_limit = max(24, len(selected))
    reference = None
    for step, image, odom in selected:
        try:
            raw = localize(memories, image)
        except (OSError, RuntimeError, ValueError) as error:
            if strict_errors:
                raise
            raw = dict(pose=None, answer=str(error), error=type(error).__name__)
            details["errors"] += 1
        details["attempted"] += 1
        current_reference = raw.get("reference_block_from_odom")
        if current_reference is not None:
            try:
                current_reference = np.asarray(current_reference, dtype=float)
                frame = np.linalg.inv(current_reference)
                valid = (current_reference.shape == (4, 4) and np.isfinite(current_reference).all()
                         and np.allclose(current_reference[3], [0, 0, 0, 1])
                         and np.allclose(current_reference[:3, :3].T @ current_reference[:3, :3], np.eye(3), atol=1e-5)
                         and np.isclose(np.linalg.det(current_reference[:3, :3]), 1.))
                if not valid or (reference is not None and not np.allclose(reference, current_reference, atol=1e-7, rtol=0)):
                    raise ValueError("localization output frame changed during initialization")
                if expected_frame is not None and not np.allclose(frame, expected_frame, atol=1e-7, rtol=0):
                    raise ValueError("localization output does not match selected anchor")
            except (ValueError, np.linalg.LinAlgError):
                details.update(reason="invalid_or_changed_output_frame")
                return None, details
            candidate.rebase(np.eye(4), target_block=str(memories[0]), target_frame=frame)
            reference = current_reference
        # Do not fit an unlabelled pose from a failed/partial worker response.
        status = candidate.add_candidate(np.asarray(odom), raw.get("pose")) if current_reference is not None else "missing_frame"
        details["localized"] += int(status == "candidate")
        if observe is not None:
            observe(step, image, odom, raw, status)
    if candidate.fit_candidates(method="huber", position_tolerance_m=position_tolerance_m):
        details.update(ready=True, reason="initialized", fit=candidate.last_fit,
                       reference_block_from_odom=reference.tolist())
        candidate.candidate_limit = INITIALIZATION_LIMIT
        return candidate, details
    details["reason"] = "no_consensus_or_fit_failed"
    return None, details


def initialize_tf(samples, memories, localize, *, observe=None, expected_frame=None, strict_errors=False, qualities=None, position_tolerance_m=1.25):
    """Retry learned-map initialization with appended frames and bounded tolerances.

    Retain the initial 24-frame policy and stop at the first successful fit.
    Other callers keep their original single-round behavior. Reference poses
    only validate coordinate frames; they never select samples or fit the TF.
    """
    if qualities is None or position_tolerance_m != 2.0:
        return _initialize_tf_round(samples, memories, localize, observe=observe, expected_frame=expected_frame,
            strict_errors=strict_errors, qualities=qualities, position_tolerance_m=position_tolerance_m)
    from .tf_frame_sampling import select_frames, extend_ladder_frames
    eligible = list(_level_station_samples(samples, limit=None))
    initial, sampling = select_frames(eligible, qualities, INITIALIZATION_LIMIT)
    by_step = {int(step): (step, image, odom) for step, image, odom in eligible}
    history = [dict(step=int(step), image=image, odom_camera=np.asarray(odom).tolist(), depth_quality=qualities.get(step, {}))
               for step, image, odom in eligible]
    lookup = {x['step']: x for x in history}
    selected = [lookup[int(x[0])] for x in initial]
    cache = {}; observed = set(); rounds = []
    def cached_localize(paths, image):
        key = str(image)
        if key not in cache: cache[key] = localize(paths, image)
        return cache[key]
    def once_observe(step, image, odom, raw, status):
        if observe is not None and step not in observed:
            observed.add(step); observe(step, image, odom, raw, status)
    for budget, tolerance in TF_RETRY_LADDER:
        if budget > 24 and history: extend_ladder_frames(selected, history, budget)
        chosen = [by_step[x['step']] for x in selected]
        candidate, details = _initialize_tf_round(samples, memories, cached_localize,
            observe=once_observe, expected_frame=expected_frame, strict_errors=strict_errors,
            position_tolerance_m=tolerance, selected_samples=chosen)
        rounds.append(dict(budget=budget, actual_frames=len(chosen), position_tolerance_m=tolerance,
            ready=details['ready'], reason=details['reason'], sample_steps=details['sample_steps'], fit=details.get('fit')))
        details.update(selection='append_only_tf_ladder_v1', sample_limit=budget, ladder_rounds=list(rounds),
            initial_sampling=sampling, unique_localization_calls=len(cache))
        if candidate is not None or details['reason']=='invalid_or_changed_output_frame':
            return candidate, details
    return None, details
