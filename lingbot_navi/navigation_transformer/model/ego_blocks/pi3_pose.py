# Pi3-derived pose head and SO(3) projection:
# Copyright (c) 2025 the authors. All rights reserved.
# See licenses/LICENSE.pi3 at the repository root for the source license and disclaimer.
# The residual block is derived via Pi3 from marepo:
# Copyright © Niantic, Inc. 2024.
# Copyright © Niantic, Inc. 2018. All rights reserved.
"""Inference adaptation of Pi3's camera head.

Source: https://github.com/yyfz/Pi3
Revision: 97ae7a43296541859d3c55eae70481f561a6789f
File: pi3/models/layers/camera_head.py
Pi3's residual-block attribution:
https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172

Modified for LingBotNavi: linear input-width adapter, variable-length token pooling,
FP32 storage/computation, optional scale readout, and separate pose outputs.
The standalone extraction omits loss and optimizer-specific code.
LingBotNavi also supplies quaternion conversion helpers for its xyzw interface.
No Pi3 pretrained weights are loaded. See NOTICE.
"""

import torch
from torch import nn
from torch.nn import functional as F


def project_to_so3(value):
    """Pi3's row normalization, transposed SVD and determinant correction."""
    with torch.autocast(device_type=value.device.type, enabled=False):
        matrix = value.float().reshape(*value.shape[:-1], 3, 3)
        u, _, vh = torch.linalg.svd(F.normalize(matrix, dim=-1).transpose(-1, -2))
        v = vh.transpose(-1, -2)
        determinant = torch.linalg.det(v @ u.transpose(-1, -2))
        corrected = torch.cat(
            (v[..., :2], v[..., 2:] * determinant[..., None, None]), dim=-1
        )
        return corrected @ u.transpose(-1, -2)


def matrix_to_quaternion(matrix):
    r = matrix.float()
    r00, r01, r02 = r[..., 0, :].unbind(-1)
    r10, r11, r12 = r[..., 1, :].unbind(-1)
    r20, r21, r22 = r[..., 2, :].unbind(-1)
    squared = torch.stack(
        (
            1 + r00 - r11 - r22,
            1 - r00 + r11 - r22,
            1 - r00 - r11 + r22,
            1 + r00 + r11 + r22,
        ),
        dim=-1,
    ).clamp_min(0)
    candidates = torch.stack(
        (
            torch.stack((squared[..., 0], r01 + r10, r02 + r20, r21 - r12), dim=-1),
            torch.stack((r01 + r10, squared[..., 1], r12 + r21, r02 - r20), dim=-1),
            torch.stack((r02 + r20, r12 + r21, squared[..., 2], r10 - r01), dim=-1),
            torch.stack((r21 - r12, r02 - r20, r10 - r01, squared[..., 3]), dim=-1),
        ),
        dim=-2,
    )
    index = squared.argmax(-1)[..., None, None].expand(*squared.shape[:-1], 1, 4)
    quaternion = F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)
    return torch.where(quaternion[..., 3:] < 0, -quaternion, quaternion)


class ResConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.res_conv1 = nn.Linear(dim, dim)
        self.res_conv2 = nn.Linear(dim, dim)
        self.res_conv3 = nn.Linear(dim, dim)

    def forward(self, features):
        residual = F.relu(self.res_conv1(features))
        residual = F.relu(self.res_conv2(residual))
        return features + F.relu(self.res_conv3(residual))


class Pi3PoseHead(nn.Module):
    """Pi3's two residual blocks -> mean pool -> two MLPs -> t3 / R9.

    Block inputs contain point tokens only, as Pi3 excludes register tokens.
    A goal query is a one-token sequence. No point truncation or padding.
    Both heads and the width adapter retain FP32 parameter/update storage.
    """

    def __init__(self, input_dim, dim=512, *, predict_scale=False):
        super().__init__()
        self.input_projection = (
            nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        )
        self.res_conv = nn.ModuleList([ResConvBlock(dim) for _ in range(2)])
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU()
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 9)
        self.predict_scale = predict_scale
        if predict_scale:
            self.fc_log_scale = nn.Linear(dim, 1)

    def _apply(self, fn, recurse=True):

        def preserve(value):
            converted = fn(value)
            if value.is_floating_point() and converted.dtype != torch.float32:
                return (
                    converted.float()
                    if value.is_meta
                    else value.to(device=converted.device, dtype=torch.float32)
                )
            return converted

        return super()._apply(preserve, recurse=recurse)

    def forward(self, features):
        """Return translation, rotation, and optionally a scalar log-scale.

        Input is [..., points, channels]; scale uses ordinary Linear initialization.
        """
        if features.ndim < 2 or features.shape[-2] == 0:
            raise ValueError("Pi3PoseHead requires a nonempty point/token sequence")
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = self.input_projection(features.float())
            for block in self.res_conv:
                features = block(features)
            pooled = self.more_mlps(features.mean(dim=-2))
            pose = (self.fc_t(pooled), project_to_so3(self.fc_rot(pooled)))
            if self.predict_scale:
                return (*pose, self.fc_log_scale(pooled).squeeze(-1))
            return pose
