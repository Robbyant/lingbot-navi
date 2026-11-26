"""Search from the last observed pose of one retained capture segment."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def run(argv):
    from examples.demo import ROOT, explore
    from examples.scene import load_scene
    from examples.visualization.frontier import render_search
    from data_toolkit.render import GaussianScene
    from data_toolkit.render.geometry import FLU_FROM_CV, quaternion_matrix
    from lingbot_navi import LingBotNavi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "examples/sample_data/scene.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "ckpts/lingbot-navi/navigation_transformer/model.safetensors")
    parser.add_argument("--gs-asset", type=Path)
    parser.add_argument("--question", default="the round dining table")
    parser.add_argument("--block", default="segment_06")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    path, scene, transform, asset = load_scene(args.scene, args.gs_asset)
    trajectory = json.loads((path.parent / scene["capture_trajectory"]).read_text())
    if trajectory["scene_id"] != scene["source_scene"] or args.block not in scene["block_ids"]:
        raise ValueError("Capture must contain the requested memory block in the same scene")
    index = scene["block_ids"].index(args.block)
    block = next(b for b in trajectory["blocks"] if b["name"] == args.block)
    cameras = np.array([trajectory["frames"][i]["camera_pose_c2w"] for i in block["indices"]])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(path.parent / scene["memory"]) as source, TemporaryDirectory() as directory:
        offsets = source["block_cu_seqlens"]
        if len(offsets) != len(scene["block_ids"]) + 1 or len(offsets) <= 2 or offsets[0] != 0:
            raise ValueError("Expected ordered multi-block scene memory")
        start, stop = offsets[index:index+2]
        xyz = source["memory_xyz"][start:stop]
        memory = Path(directory) / "partial.npz"
        np.savez(memory, memory_xyz=xyz, memory_features=source["memory_features"][start:stop],
                 block_cu_seqlens=offsets[index:index+2] - start)
        view = cameras[-1].copy()
        view[:3, :3] = view[:3, :3] @ FLU_FROM_CV
        renderer = GaussianScene(asset, args.device)
        image = args.output_dir / "current_camera.png"
        renderer.render(view, np.array(trajectory["intrinsics"]), trajectory["width"],
                        trajectory["height"]).rgb.save(image)
        del renderer
        model = LingBotNavi.from_pretrained(args.weights, device=args.device)
        localization = model.predict(memory, "Where am I?", image=image, task="localization")
        if localization.get("finish_reason") != "eos" or len(localization["predicted_poses"]) != 1:
            raise ValueError("Could not localize the current camera in the retained block")
        pose = localization["predicted_poses"][0]
        current = pose[:3]
        local_camera = np.eye(4)
        local_camera[:3, :3] = quaternion_matrix(pose[3:])
        local_camera[:3, 3] = current
        transform = cameras[-1] @ np.linalg.inv(local_camera)
        result = explore(model, memory, args.question, current)
        del model
    result["asset_from_retained_block_estimate"] = transform.tolist()
    result["camera_localization"] = localization
    result["retained_blocks"] = [block["name"]]
    result["query_frame_index"] = block["indices"][-1]
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["exploration"]["status"] == "frontier":
        render_search(args.output_dir, args.question, result, cameras, trajectory, transform,
                      scene["visualization"], asset, args.device, path.parent / "coverage.npz")
