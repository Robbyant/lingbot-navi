import torch
from torch import nn


class MemoryPointHead(nn.Module):
    def __init__(self, width, hidden=384):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.mlp(self.norm(hidden.float()))


def transform_memory_points(
    predictions, block_poses, block_cu, scene_cu, *, inverse=False
):
    """Transform packed points with each chunk's local-to-first-block SE(3).

    Row-vector convention: global = local @ R.T + t. Used for GT frame
    conversion and optional reconstruction export.
    """
    blocks = block_cu.detach().cpu().tolist()
    scenes = scene_cu.detach().cpu().tolist()
    result = {}
    for scene, (a, b) in enumerate(zip(scenes[:-1], scenes[1:])):
        if len(block_poses[scene]) != b - a:
            raise ValueError("one pose is required per chunk")
        chunks, offset = ([], 0)
        with torch.autocast(device_type=predictions[scene].device.type, enabled=False):
            for j, block in enumerate(range(a, b)):
                count = blocks[block + 1] - blocks[block]
                points = predictions[scene][offset : offset + count].float()
                t, r = block_poses[scene][j][:2]
                t, r = (t.float(), r.float())
                chunks.append((points - t) @ r if inverse else points @ r.T + t)
                offset += count
        if offset != len(predictions[scene]):
            raise ValueError("point count differs from chunk offsets")
        result[scene] = torch.cat(chunks)
    return result
