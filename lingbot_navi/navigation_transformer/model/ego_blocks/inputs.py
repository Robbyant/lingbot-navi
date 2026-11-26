import itertools
import torch
from torch import Tensor


def offsets(value: Tensor, total: int, name: str) -> tuple[int, ...]:
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a 1-D integer cumulative-length tensor")
    result = tuple(value.detach().cpu().tolist())
    if len(result) < 2 or result[0] != 0 or result[-1] != total:
        raise ValueError(f"{name} must start at 0 and end at {total}")
    if any((a >= b for a, b in itertools.pairwise(result))):
        raise ValueError(f"{name} cannot contain empty groups")
    return result


def image_span(mask: Tensor, *, allow_empty: bool = False) -> tuple[int, int]:
    """Return the inclusive-wrapper current-image span in an unpadded prompt."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("current_image_mask must be boolean")
    indices = mask.nonzero().flatten()
    if not indices.numel() and allow_empty:
        return (0, 0)
    if indices.numel() < 3:
        raise ValueError(
            "current image must include vision_start, visual tokens and vision_end"
        )
    start, end = (int(indices[0]), int(indices[-1]) + 1)
    if end - start != indices.numel():
        raise ValueError("current_image_mask must select one contiguous image span")
    return (start, end)
