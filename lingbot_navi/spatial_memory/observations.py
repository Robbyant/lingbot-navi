from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ObservationBatch:
    features: torch.Tensor
    positions: torch.Tensor
    rgb: torch.Tensor | None
    depths: torch.Tensor
    before_point_filter: int
    after_point_filter: int


def unproject_patches(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pose: torch.Tensor,
    *,
    patch_height: int | None = None,
    patch_width: int | None = None,
    input_height: int | None = None,
    input_width: int | None = None,
    content_height: int | None = None,
    content_width: int | None = None,
    padding_top: int = 0,
    padding_left: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject vision-patch centres in the Robby FLU camera convention.

    Depth is positive along camera +X, image right is -Y, and image down is
    -Z.  ``pose`` is camera-to-world in the same FLU convention.
    """

    if depth.ndim != 2 or intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError(
            "expected depth HxW, intrinsic 3x3 and camera pose 4x4; got "
            f"{tuple(depth.shape)}, {tuple(intrinsic.shape)}, {tuple(pose.shape)}"
        )
    if None in (patch_height, patch_width, input_height, input_width):
        raise ValueError("provide patch and canvas height/width")
    patch_height = int(patch_height)
    patch_width = int(patch_width)
    input_height = int(input_height)
    input_width = int(input_width)
    if min(patch_height, patch_width, input_height, input_width) <= 0:
        raise ValueError("patch and input dimensions must be positive")

    height, width = depth.shape
    patch_rows = input_height // patch_height
    patch_columns = input_width // patch_width
    if patch_rows <= 0 or patch_columns <= 0:
        raise ValueError("patch dimensions must not exceed input dimensions")
    row_centres = (
        torch.arange(patch_rows, dtype=depth.dtype, device=depth.device) + 0.5
    ) * float(patch_height)
    column_centres = (
        torch.arange(patch_columns, dtype=depth.dtype, device=depth.device) + 0.5
    ) * float(patch_width)
    content_height = input_height if content_height is None else int(content_height)
    content_width = input_width if content_width is None else int(content_width)
    if (
        min(content_height, content_width) <= 0
        or min(padding_top, padding_left) < 0
        or padding_top + content_height > input_height
        or padding_left + content_width > input_width
    ):
        raise ValueError("content must fit within the padded input")
    u = (column_centres - padding_left) * (float(width) / content_width)
    v = (row_centres - padding_top) * (float(height) / content_height)
    u_index = u.long().clamp(0, width - 1)
    v_index = v.long().clamp(0, height - 1)
    patch_depth = depth[v_index][:, u_index]
    inside = ((v >= 0) & (v < height))[:, None] & ((u >= 0) & (u < width))[None, :]
    patch_depth = torch.where(inside, patch_depth, 0.0)

    u_grid = u.unsqueeze(0).expand(patch_rows, -1)
    v_grid = v.unsqueeze(1).expand(-1, patch_columns)
    x_camera = patch_depth
    normalized_image_y = (v_grid - intrinsic[1, 2]) / intrinsic[1, 1]
    # K may contain upper-triangular pixel skew.  Solve the calibrated
    # pinhole equations before converting OpenCV image-right/down to FLU
    # camera-left/up; ignoring K[0,1] shifts every off-centre 3D point.
    image_right = (
        (u_grid - intrinsic[0, 2] - intrinsic[0, 1] * normalized_image_y)
        * patch_depth
        / intrinsic[0, 0]
    )
    y_camera = -image_right
    z_camera = -normalized_image_y * patch_depth
    camera_points = torch.stack((x_camera, y_camera, z_camera), dim=-1)
    world_points = (
        torch.einsum("ij,hwj->hwi", pose[:3, :3], camera_points) + pose[:3, 3]
    )
    return world_points, patch_depth


def prepare_observations(
    patch_features: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pose: torch.Tensor,
    patch_rgb: torch.Tensor | None,
    *,
    min_depth: float,
    max_depth: float,
    patch_height: int | None = None,
    patch_width: int | None = None,
    input_height: int | None = None,
    input_width: int | None = None,
    content_height: int | None = None,
    content_width: int | None = None,
    padding_top: int = 0,
    padding_left: int = 0,
) -> ObservationBatch:
    """Unproject one calibrated feature grid and retain valid axial depths."""

    features = torch.as_tensor(patch_features, dtype=torch.float32).detach().to("cpu")
    depth = torch.as_tensor(depth, dtype=torch.float32).detach().to("cpu")
    intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32).detach().to("cpu")
    pose = torch.as_tensor(pose, dtype=torch.float32).detach().to("cpu")
    rgb = (
        torch.as_tensor(patch_rgb, dtype=torch.float32).detach().to("cpu")
        if patch_rgb is not None
        else None
    )
    if features.ndim != 3:
        raise ValueError(f"patch features must be HxWxD, got {tuple(features.shape)}")
    feature_height, feature_width, feature_dim = features.shape
    if rgb is not None and rgb.shape != (feature_height, feature_width, 3):
        raise ValueError(
            "patch_rgb must align with features; got "
            f"{tuple(rgb.shape)} vs {(feature_height, feature_width, 3)}"
        )

    positions, patch_depth = unproject_patches(
        depth,
        intrinsic,
        pose,
        patch_height=patch_height,
        patch_width=patch_width,
        input_height=input_height,
        input_width=input_width,
        content_height=content_height,
        content_width=content_width,
        padding_top=padding_top,
        padding_left=padding_left,
    )
    if positions.shape[:2] != (feature_height, feature_width):
        raise ValueError("feature grid and calibrated depth-patch grid differ")
    clip_distance = patch_depth
    valid_depth = (
        (patch_depth > 0.0)
        & (clip_distance > float(min_depth))
        & (clip_distance < float(max_depth))
        & torch.isfinite(patch_depth)
        & torch.isfinite(clip_distance)
        & torch.isfinite(positions).all(dim=-1)
    )
    before = int(valid_depth.sum().item())
    keep = valid_depth
    after = int(keep.sum().item())
    flat_mask = keep.reshape(-1)
    return ObservationBatch(
        features=features.reshape(-1, feature_dim)[flat_mask],
        positions=positions.reshape(-1, 3)[flat_mask],
        rgb=rgb.reshape(-1, 3)[flat_mask] if rgb is not None else None,
        depths=clip_distance.reshape(-1)[flat_mask],
        before_point_filter=before,
        after_point_filter=after,
    )
