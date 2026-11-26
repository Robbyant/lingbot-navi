from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FeedbackPolicy:
    xyz_decimals: int = 2
    quaternion_decimals: int = 4


def normalize_feedback_mode(mode: str) -> str:
    normalized = str(mode).lower().replace("_", "").replace("-", "")
    modes = {"text": "text", "pe": "pe", "hiddenstate": "hidden_state"}
    if normalized not in modes:
        raise ValueError("feedback mode must be text, pe, or hidden_state")
    return modes[normalized]


def canonical_pose(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().clone()
    if value.shape != (7,) or not bool(torch.isfinite(value).all()):
        raise ValueError("pose feedback requires finite XYZ + xyzw")
    norm = value[3:].norm()
    if norm < 1e-08:
        raise ValueError("cannot format a zero quaternion")
    value[3:] /= norm
    for i in (6, 3, 4, 5):
        if value[i] != 0:
            if value[i] < 0:
                value[3:] *= -1
            break
    return value


def format_feedback(kind: str, value: torch.Tensor, policy: FeedbackPolicy) -> str:
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("coordinate feedback must be finite")

    def number(v, digits):
        rounded = round(float(v), digits)
        return f"{(0.0 if rounded == 0 else rounded):+.{digits}f}"

    if kind == "pose":
        value = canonical_pose(value)
    elif kind != "center" or value.shape != (3,):
        raise ValueError("feedback must be a center [3] or pose [7]")
    xyz = ",".join((number(v, policy.xyz_decimals) for v in value[:3]))
    if kind == "center":
        return f" at ({xyz})"
    quat = ",".join((number(v, policy.quaternion_decimals) for v in value[3:]))
    return f" at ({xyz}), quaternion_xyzw=({quat})"
