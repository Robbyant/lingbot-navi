"""Rank a complete 30-degree panorama by mean valid metric depth."""
import math
import numpy as np

def depth_openness(depth):
    values = np.asarray(depth, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.mean(valid, dtype=np.float64)) if valid.size else -math.inf

def turn_between_views(source, destination, turn):
    delta = (destination - source + 6) % 12 - 6
    return all((turn('turn_left' if delta > 0 else 'turn_right') for _ in range(abs(delta))))

def select_observation_view(read_depth, turn, *, turn_deg=30.0):
    if not math.isclose(turn_deg, 30.0):
        raise ValueError('observation view selection requires 30 degree turns')
    scores = [depth_openness(read_depth())]
    for _ in range(11):
        if not turn('turn_left'):
            return None
        scores.append(depth_openness(read_depth()))
    ranked = sorted(range(12), key=lambda i: -scores[i])[:5]
    if not turn_between_views(11, ranked[0], turn):
        return None
    return {'bins': ranked, 'depths_m': [s if math.isfinite(s) else None for s in scores]}
