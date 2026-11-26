"""Save one synthetic localization query using the shared perspective renderer."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from data_toolkit.render import CameraSpec, GaussianScene, PERSPECTIVE_RENDERER
from examples.scene import load_scene


def render_query_view(renderer, view):
    """Render the capture camera, then update calibration through resize/crop."""
    capture = view.get("capture")
    camera = capture or view
    calibration = CameraSpec(camera["width"], camera["height"], camera["K"])
    rgb = renderer.render_camera(
        calibration, np.array(view["camera_to_world_cv"])
    ).rgb
    if capture:
        width, height = rgb.size
        longest = max(width, height)
        target = min(longest, capture["long_side"])
        resized = (width * target // longest, height * target // longest)
        rgb = rgb.resize(resized, resample=Image.Resampling.LANCZOS)
        calibration = calibration.resized(*resized)
        alignment = capture["alignment"]
        aligned = tuple(side // alignment * alignment for side in resized)
        left, top = ((resized[i] - aligned[i]) // 2 for i in range(2))
        rgb = rgb.crop((left, top, left + aligned[0], top + aligned[1]))
        calibration = calibration.cropped(left, top, *aligned)
    return rgb, calibration.K


def capture_query(scene_file, output_dir, *, view_index=0, gs_asset=None, device="cuda:0"):
    scene_path, scene, _, asset = load_scene(scene_file, gs_asset=gs_asset)
    views = scene.get("views", [])
    if not 0 <= view_index < len(views):
        raise ValueError("--view must select an existing scene.views camera")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Query directory already exists; choose a new output directory")
    view = views[view_index]
    renderer = GaussianScene(asset, device=device)
    try:
        rgb, k = render_query_view(renderer, view)
        view.update(width=rgb.width, height=rgb.height, K=k.tolist())
    finally:
        renderer.close()
    output.mkdir(parents=True, exist_ok=False)
    image = output / "current.png"
    rgb.save(image)
    # Keep the original coordinate frame and memory, with explicit absolute paths.
    # The reference pose is for rendering only; the inference model reads RGB.
    for saved in views:
        if saved.get("image"):
            saved["image"] = str((scene_path.parent / saved["image"]).resolve())
    view["image"] = str(image)
    scene["views"] = [view] + [v for i, v in enumerate(views) if i != view_index]
    scene["memory"] = str((scene_path.parent / scene["memory"]).resolve())
    scene["gs_asset"] = str(asset)
    scene["query_rendering"] = {
        "renderer": PERSPECTIVE_RENDERER,
        "source_scene": str(scene_path),
        "source_view_index": view_index,
        "image": str(image),
        "background": "black",
        "capture": view.get("capture"),
        "output_size": list(rgb.size),
        "output_K": k.tolist(),
    }
    (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return image, output / "scene.json"
