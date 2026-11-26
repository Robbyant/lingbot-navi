"""Run one LingBot-Navi question and save its optional scene visualizations."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]


def add_render_options(parser):
    parser.add_argument(
        "--gs-asset",
        type=Path,
        help="Override with a full-quality Gaussian PLY in the same asset coordinate frame",
    )
    parser.add_argument(
        "--bev-z-min", type=float, help="Minimum Gaussian-center Z in asset meters"
    )
    parser.add_argument(
        "--bev-z-max", type=float, help="Maximum Gaussian-center Z in asset meters"
    )
    parser.add_argument(
        "--bev-bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX")
    )
    parser.add_argument(
        "--render-size", type=int, default=1024, help="Square BEV resolution"
    )



def explore(model, memory, question, current_xyz, *, image=None, **options):
    """Retrieve a target, then ask for one frontier only when retrieval says absent."""
    import math
    from lingbot_navi.navigation_transformer.prompts import (
        ABSENT, IMAGE_ABSENT, canonical_goal, frontier_messages, make_messages,
    )
    reported_xyz = canonical_goal(current_xyz)
    task = "localization" if image is not None else "grounding"
    retrieval = model.predict(memory, question, image=image, task=task, **options)
    answer = retrieval["generated_text"].replace("<|im_end|>", "").strip()
    if retrieval.get("finish_reason") != "eos":
        raise ValueError("Target retrieval did not finish normally")
    absent = IMAGE_ABSENT if image is not None else ABSENT
    history = make_messages(question, image=image, task=task)
    history.append({"role": "assistant", "content": answer})
    if answer == absent:
        if any(retrieval.get(k) for k in ("predicted_centers", "predicted_poses", "predicted_viewpoints", "trajectory")):
            raise ValueError("Absent retrieval also returned a spatial target")
        continuation = frontier_messages(current_xyz, image_target=image is not None, absence=answer)
        result = model.predict(memory, question, image=image, task=task,
                               continuation_messages=continuation, **options)
        centers = result.get("predicted_centers") or []
        if (result.get("finish_reason") != "eos" or len(centers) != 1
                or len(centers[0]) != 3 or not all(math.isfinite(v) for v in centers[0])
                or any(result.get(k) for k in ("predicted_poses", "predicted_viewpoints", "trajectory"))):
            raise ValueError("Exploration must return exactly one finite frontier position")
        history.extend([continuation[1], {"role": "assistant", "content": result["generated_text"]}])
        status = "frontier"
    else:
        targets = retrieval.get("predicted_poses" if image is not None else "predicted_centers")
        if not targets:
            raise ValueError("Retrieval returned neither a target nor the expected absence answer")
        result, status = retrieval, "target_found"
    if result.get("coordinate_frame") != "first_block_meters":
        raise ValueError("Exploration requires first-block coordinates")
    result = dict(result)
    result["exploration"] = dict(status=status, current_xyz=reported_xyz,
                                target_kind="image" if image is not None else "language",
                                retrieval=retrieval, messages=history)
    return result


def main():
    if sys.argv[1:2] == ["partial-exploration"]:
        from examples.partial_exploration import run
        return run(sys.argv[2:])
    if sys.argv[1:2] == ["execute-navigation"]:
        return execute_navigation(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="LingBot-Navi single-question spatial inference",
        epilog="Run `python -m examples.demo execute-navigation --help` for local route execution.",
    )
    parser.add_argument("--weights", type=Path, help="Defaults to the release checkpoint under ckpts/lingbot-navi/navigation_transformer/")
    parser.add_argument("--memory", default=ROOT / "examples/sample_data/memory.npz", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--task",
        choices=("qa", "grounding", "localization", "coordinate", "exploration"),
        default="qa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--goal-xyz", type=float, nargs=3)
    parser.add_argument("--current-xyz", type=float, nargs=3,
                        help="Exploration: current position in the first memory block; --image is the target image")
    parser.add_argument(
        "--localization-result",
        type=Path,
        help="Previous localization JSON for coordinate continuation",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Also return block and point predictions",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON output file; default prints to stdout"
    )
    parser.add_argument("--scene", type=Path, help="Scene JSON paired with this memory")
    parser.add_argument(
        "--output-dir", "--render-dir", dest="render_dir",
        type=Path,
        help="Save the answer and visualizations here; default: repository outputs/<run>/<task>",
    )
    parser.add_argument(
        "--view", type=int,
        help="For localization, generate the query from this saved scene camera",
    )

    add_render_options(parser)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = ROOT / "ckpts/lingbot-navi/navigation_transformer/model.safetensors"
    if args.image is None and args.view is None and args.task in ("localization", "coordinate"):
        args.image = ROOT / "examples/sample_data/query.png"
    if args.scene is None and (args.gs_asset or args.render_dir or args.view is not None):
        args.scene = ROOT / "examples/sample_data/scene.json"
    if args.view is not None:
        if args.task != "localization" or not args.scene or args.image:
            parser.error("--view requires localization and --scene, without --image")
    if args.scene and args.render_dir is None:
        run_name = datetime.now().strftime("demo-%Y%m%d-%H%M%S-%f")
        task_name = "navigation" if args.task == "coordinate" else args.task
        args.render_dir = Path(__file__).resolve().parents[1] / "outputs" / run_name / task_name
    if args.render_dir:
        args.render_dir = args.render_dir.expanduser().resolve()
    if args.render_dir:
        from examples.scene import load_scene
        from data_toolkit.render import require_backend

        scene_path, scene, _, _ = load_scene(args.scene, gs_asset=args.gs_asset)
        if (
            args.memory.expanduser().resolve()
            != (scene_path.parent / scene["memory"]).resolve()
        ):
            parser.error("--memory must match the memory paired with --scene")
        if args.render_dir.exists():
            parser.error("Output directory already exists; choose a new directory")
        if not 256 <= args.render_size <= 4096:
            parser.error("--render-size must be between 256 and 4096")
        require_backend(args.device)
    if args.output and args.output.exists():
        parser.error("Output already exists")
    if args.current_xyz is not None and args.task != "exploration":
        parser.error("--current-xyz is only for exploration")
    if args.task == "exploration" and args.goal_xyz is not None:
        parser.error("Exploration predicts its own frontier; do not supply --goal-xyz")
    continuation = {}
    if args.localization_result:
        prior = json.loads(args.localization_result.read_text())
        if not prior.get("predicted_poses"):
            parser.error("Localization result has no pose")
        if args.task == "exploration":
            if args.current_xyz is not None:
                parser.error("Use either --current-xyz or --localization-result for exploration")
            if prior.get("coordinate_frame") != "first_block_meters":
                parser.error("Localization must use the same first-block frame as this memory")
            args.current_xyz = prior["predicted_poses"][0][:3]
        else:
            continuation = dict(
                current_pose=prior["predicted_poses"][0],
                localization_answer=prior["generated_text"],
            )
    if args.task == "exploration" and args.current_xyz is None:
        parser.error("Exploration requires --current-xyz or --localization-result from the same memory")
    if args.view is not None:
        from examples.visualization.capture import capture_query

        query_dir = args.render_dir.with_name(args.render_dir.name + "-query")
        args.image, args.scene = capture_query(
            args.scene, query_dir, view_index=args.view,
            gs_asset=args.gs_asset, device=args.device,
        )
        print(f"Query image: {args.image}", flush=True)

    from lingbot_navi import LingBotNavi

    model = LingBotNavi.from_pretrained(
        args.weights,
        device=args.device,
    )
    if args.task == "exploration":
        result = explore(model, args.memory, args.question, args.current_xyz,
                         image=args.image, max_new_tokens=args.max_new_tokens,
                         return_geometry=args.geometry)
    else:
        result = model.predict(
            args.memory,
            args.question,
            image=args.image,
            task=args.task,
            max_new_tokens=args.max_new_tokens,
            goal_xyz=args.goal_xyz,
            return_geometry=args.geometry,
            **continuation,
        )
    # Release the model before saving the demo visualizations.
    if args.render_dir:
        import gc
        import torch

        del model
        gc.collect()
        torch.cuda.empty_cache()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    if args.render_dir:
        from examples.visualization import render_result

        render_result(
            result,
            args.scene,
            args.render_dir,
            image=args.image,
            goal_xyz=args.goal_xyz,
            device=args.device,
            gs_asset=args.gs_asset,
            z_min=args.bev_z_min,
            z_max=args.bev_z_max,
            bounds=args.bev_bounds,
            size=args.render_size,
        )
        print(f"Demo results: {args.render_dir}", flush=True)


def _rollout(args, ffmpeg):
    from tempfile import TemporaryDirectory
    from math import cos, sin
    import numpy as np
    from scipy.spatial.transform import Rotation

    from data_toolkit.render import CameraSpec, GaussianScene
    from data_toolkit.render.geometry import (
        FLU_FROM_CV, points, rigid_transform, transform_points,
    )
    from examples.visualization.execution import execution_video
    from examples.scene import load_scene
    from lingbot_navi.local_planner import LocalPlanner, LocalPlannerOptions
    from lingbot_navi.local_planner.controller import MOVE_FORWARD, STOP, TURN_LEFT, TURN_RIGHT

    scene_path, scene, asset_from_block, asset = load_scene(args.scene, gs_asset=args.gs_asset)
    result = json.loads(args.route_result.read_text())
    if result.get("coordinate_frame") != "first_block_meters":
        raise ValueError("Demo 4 result must be in the first-block coordinate frame")
    route = transform_points(points(result.get("trajectory") or []), asset_from_block)
    if len(route) < 2:
        raise ValueError("Demo 4 result must contain a predicted route with at least two points")
    views = scene.get("views", [])
    if not 0 <= args.view < len(views):
        raise ValueError("--view must index a saved scene camera")
    view = views[args.view]
    camera = CameraSpec(view["width"], view["height"], view["K"])
    initial_cv = rigid_transform(view["camera_to_world_cv"])
    position = initial_cv[:3, 3].copy()
    rotation = Rotation.from_matrix(initial_cv[:3, :3] @ FLU_FROM_CV.T)

    # Use the scene's saved query camera as simulator ground truth. Keep the
    # predicted route in the asset frame without shifting it to that camera.
    options = LocalPlannerOptions()
    session = LocalPlanner.from_pretrained(
        args.local_weights, args.vision_model, device=args.device, options=options,
    ).start(route)
    settings = scene.get("visualization", {})
    floor_z = float(settings.get("floor_z", 0.0))
    actions = {STOP: "Stop", MOVE_FORWARD: "Forward", TURN_LEFT: "Turn left", TURN_RIGHT: "Turn right"}
    steps = []
    executed = []

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent) as temporary, GaussianScene(asset, args.device) as renderer:
        stage = Path(temporary) / "execution"
        stage.mkdir()
        with execution_video(
            renderer, camera, route, position, floor_z, stage / "execution.mp4",
            ffmpeg=ffmpeg, fps=1 / options.observation_interval_s,
        ) as write_frame:
            for index in range(args.max_steps + 1):
                flu = rotation.as_matrix()
                c2w = np.eye(4)
                c2w[:3, :3], c2w[:3, 3] = flu @ FLU_FROM_CV, position
                rgb = renderer.render_camera(camera, c2w).rgb
                executed.append(position.copy())
                remaining = float(np.linalg.norm(position[:2] - route[-1, :2]))
                if index == args.max_steps:
                    reason, action_label = "step_limit", "Step limit"
                else:
                    step = session.step(rgb, position=position, rotation_xyzw=rotation.as_quat())
                    action = step.decision.action
                    action_label = actions[action]
                    reason = "planner_stop" if action == STOP else None
                    steps.append(step.to_dict())

                write_frame(rgb, index, action_label, remaining, executed, flu[:, 0])
                if reason:
                    break

                diagnostics = step.decision.diagnostics
                rotation = Rotation.from_euler("z", diagnostics["target_heading_deg"], degrees=True) * rotation
                if action == MOVE_FORWARD:
                    forward = rotation.as_matrix()[:, 0]
                    yaw = np.arctan2(forward[1], forward[0])
                    position += diagnostics["move_distance_m"] * np.array([cos(yaw), sin(yaw), 0.0])

        (stage / "execution.json").write_text(json.dumps(dict(
            coordinate_frame="asset_world_meters_z_up", view_index=args.view,
            collision_model="none", stop_reason=reason, final_distance_to_route_end_xy_m=remaining,
            planned_route=route.tolist(), executed_positions=np.asarray(executed).tolist(),
            steps=steps,
        ), indent=2, allow_nan=False) + "\n")
        stage.rename(args.output_dir)
    print(f"{args.output_dir / 'execution.mp4'} | {reason} | {remaining:.2f} m to route end", flush=True)


def execute_navigation(argv):
    import shutil

    parser = argparse.ArgumentParser(
        prog="python -m examples.demo execute-navigation",
        description="Execute the Demo 4 route with the learned local planner in 3DGS.",
    )
    parser.add_argument("--route-result", type=Path, required=True, help="Demo 4 result.json")
    parser.add_argument("--scene", type=Path, default=ROOT / "examples/sample_data/scene.json")
    parser.add_argument("--gs-asset", type=Path)
    parser.add_argument("--local-weights", type=Path, default=ROOT / "ckpts/lingbot-navi/local_planner/checkpoint.pt")
    parser.add_argument("--vision-model", type=Path, default=ROOT / "ckpts/lingbot-navi/navigation_transformer", help="Navigation checkpoint directory containing the frozen visual encoder")
    parser.add_argument("--view", type=int, default=0, help="Saved scene camera used as the simulator start")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("Install FFmpeg with the libx264 encoder to write the MP4")
    _rollout(args, ffmpeg)


if __name__ == "__main__":
    main()
