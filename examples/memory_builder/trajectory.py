"""Saved capture poses and explicit block membership."""

import json
from pathlib import Path
from data_toolkit.render.camera import CameraSpec
from data_toolkit.render.geometry import rigid_transform


def validate_blocks(blocks, frame_count):
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("provide an ordered, nonempty blocks list")
    names, used = set(), set()
    for block in blocks:
        name, indices = block.get("name"), block.get("indices")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in names
        ):
            raise ValueError("block names must be unique directory names")
        if (
            not isinstance(indices, list)
            or len(indices) < 8
            or any(type(i) is not int or not 0 <= i < frame_count for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))
        ):
            raise ValueError("each block needs at least eight increasing frame indices")
        names.add(name)
        used.update(indices)
    if used != set(range(frame_count)):
        raise ValueError("every capture frame must belong to a block")
    return blocks


def load_trajectory(path):
    data = json.loads(Path(path).read_text())
    if (
        data.get("format") != "lingbot-navi-capture-trajectory-v1"
        or data.get("camera_axes") != "FLU"
        or data.get("pose_type") != "camera_to_world"
        or data.get("units") != "meter"
    ):
        raise ValueError("expected metric camera-to-world FLU capture poses")
    scene = data.get("scene_id")
    if not isinstance(scene, str) or not scene or "/" in scene or "\\" in scene:
        raise ValueError("invalid scene_id")
    CameraSpec(data["width"], data["height"], data["intrinsics"])
    for frame in data["frames"]:
        rigid_transform(frame["camera_pose_c2w"])
    validate_blocks(data["blocks"], len(data["frames"]))
    return data
