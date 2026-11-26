"""Read the scene metadata paired with a demo memory."""

import json
from pathlib import Path
from data_toolkit.render.geometry import rigid_transform


def load_scene(path, gs_asset=None):
    path = Path(path).expanduser().resolve()
    scene = json.loads(path.read_text())
    transform = rigid_transform(scene["asset_from_first_block"])
    if scene.get("coordinate_frame") != "first_block_meters":
        raise ValueError("Scene must map the first-block-meters model frame")
    asset_path = (
        Path(gs_asset).expanduser().resolve()
        if gs_asset is not None
        else (path.parent / scene["gs_asset"]).resolve()
    )
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Gaussian asset not found: {asset_path}. Download it separately and pass --gs-asset /path/to/scene.ply."
        )
    return path, scene, transform, asset_path
