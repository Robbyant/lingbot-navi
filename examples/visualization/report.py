"""Save a single answer as a cutaway BEV and deterministic perspective renders."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from data_toolkit.render.geometry import (
    points,
    quaternion_matrix,
    FLU_FROM_CV,
    rigid_transform,
    transform_points,
    look_at,
    bev_camera,
    bev_pixels,
    pinhole_pixels,
)
from data_toolkit.render import GaussianScene, PERSPECTIVE_RENDERER

from examples.scene import load_scene
from .drawing import RED, BLUE, GREEN, INK, font, label, dot, arrow, card


def pose_camera(pose, asset_from_block):
    """Model pose is camera-to-first-block, FLU axes, xyzw quaternion."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected [x,y,z,qx,qy,qz,qw]")
    flu = np.eye(4)
    flu[:3, :3] = quaternion_matrix(pose[3:])
    flu[:3, 3] = pose[:3]
    world = asset_from_block @ flu
    world[:3, :3] = world[:3, :3] @ FLU_FROM_CV
    return world


def observation_candidates(renderer, views, target):
    """Re-aim saved camera positions; prefer unoccluded views at 0.75--6 meters.

    Depth/alpha give a heuristic visibility test, not a navigation or collision proof.
    All candidates are retained in the report for reproducibility.
    """
    candidates = []
    probe_k = np.array([[140.0, 0.0, 160.0], [0.0, 140.0, 120.0], [0.0, 0.0, 1.0]])
    for i, view in enumerate(views):
        position = rigid_transform(view["camera_to_world_cv"])[:3, 3]
        distance = float(np.linalg.norm(target - position))
        if distance < 0.05:
            continue
        camera = look_at(position, target)
        observation = renderer.render(camera, probe_k, 320, 240)
        depth, alpha = observation.depth, observation.alpha
        region = np.s_[116:125, 156:165]
        supported = alpha[region] >= 0.5
        foreground = supported & (depth[region] < distance - 0.35)
        blocked_fraction = float(foreground.mean())
        coverage = float(supported.mean())
        in_range = 0.75 <= distance <= 6.0
        visible = blocked_fraction <= 0.25 and coverage >= 0.5
        # Visibility is the first criterion; then distance range and ideal 2.5 m.
        rank = (not visible, not in_range, blocked_fraction, abs(distance - 2.5), i)
        candidates.append(
            dict(
                saved_view_index=i,
                distance_m=distance,
                in_distance_range=in_range,
                likely_visible=visible,
                foreground_fraction=blocked_fraction,
                center_coverage=coverage,
                camera_to_asset_cv=camera.tolist(),
                rank=list(rank),
            )
        )
    if not candidates:
        raise ValueError(
            "Object rendering requires at least one noncoincident saved camera"
        )
    return min(candidates, key=lambda x: tuple(x["rank"])), candidates


def _result_arrays(result, transform):
    if result.get("coordinate_frame") != "first_block_meters":
        raise ValueError("Result must declare coordinate_frame=first_block_meters")
    centers = points(result.get("predicted_centers", []))
    poses = points(result.get("predicted_poses", []), 7)
    path_value = result.get("trajectory")
    route = points(path_value if path_value is not None else [])
    if len(route) == 1:
        raise ValueError("A trajectory must contain at least two points")
    return (
        centers,
        transform_points(centers, transform),
        poses,
        [pose_camera(p, transform) for p in poses],
        transform_points(route, transform),
    )


def render_result(
    result,
    scene_path,
    output_dir,
    *,
    image=None,
    goal_xyz=None,
    device="cuda:0",
    gs_asset=None,
    z_min=None,
    z_max=None,
    bounds=None,
    size=1024,
):
    """Render one existing answer; no model or data-loader construction.

    The output directory must be new. Asset-world Z is up; default cutaway
    retains Gaussian centers from floor - 0.1 m to floor + 1.6 m.
    """
    if not isinstance(size, int) or not 256 <= size <= 4096:
        raise ValueError("size must be an integer between 256 and 4096")
    scene_path, scene, transform, asset_path = load_scene(scene_path, gs_asset=gs_asset)
    centers, world_centers, poses, cameras, route = _result_arrays(result, transform)
    exploration = result.get("exploration")
    frontier = None
    current = None
    if exploration:
        current = transform_points(points([exploration["current_xyz"]]), transform)[0]
        if exploration["status"] == "frontier":
            frontier = world_centers[0]
            # A frontier is not an observed object: no invented object-view renders.
            centers, world_centers = points([]), points([])
    goal = (
        transform_points(points([goal_xyz]), transform)[0]
        if goal_xyz is not None
        else None
    )
    views = scene.get("views", [])
    if len(centers) and not views:
        raise ValueError("Object rendering needs scene.views camera positions")
    if image is not None:
        with Image.open(image) as im:
            input_image = ImageOps.exif_transpose(im).convert("RGB")
    else:
        input_image = None
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError("Render directory already exists; choose a new directory")
    settings = scene.get("visualization", {})
    floor_z = float(settings.get("floor_z", 0.0))
    z_min = float(z_min if z_min is not None else floor_z - 0.1)
    z_max = float(z_max if z_max is not None else floor_z + 1.6)
    if not np.isfinite([z_min, z_max]).all() or z_min >= z_max:
        raise ValueError("BEV Z limits must be finite and increasing")
    renderer = GaussianScene(asset_path, device)
    selected = (renderer.xyz[:, 2] >= z_min) & (renderer.xyz[:, 2] <= z_max)
    if not selected.any():
        raise ValueError("The BEV slice contains no Gaussian centers")
    if bounds is None:
        bounds = settings.get("bev_bounds")
    if bounds is None:
        xy = renderer.xyz[selected, :2]
        low, high = np.quantile(xy, [0.01, 0.99], axis=0)
        extras = [
            world_centers[:, :2],
            route[:, :2],
            np.array([c[:2, 3] for c in cameras]).reshape(-1, 2),
        ]
        if goal is not None:
            extras.append(goal[None, :2])
        for position in (current, frontier):
            if position is not None:
                extras.append(position[None, :2])
        if len(centers):
            extras.append(
                np.array(
                    [rigid_transform(v["camera_to_world_cv"])[:2, 3] for v in views]
                )
            )
        extra = np.concatenate(extras)
        if len(extra):
            low, high = np.minimum(low, extra.min(0)), np.maximum(high, extra.max(0))
        bounds = [*(low - 0.5), *(high + 0.5)]
    bev_c2w, bev_k, bounds = bev_camera(bounds, z_max + 3.0, size)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if exploration:
        (output_dir / "conversation.json").write_text(
            json.dumps(exploration["messages"], indent=2, ensure_ascii=False) + "\n")
    bev = renderer.render(
        bev_c2w, bev_k, size, size, ortho=True, z_range=(z_min, z_max)
    ).rgb
    bev.save(output_dir / "bev_base.png")
    overlay = bev.copy()
    draw = ImageDraw.Draw(overlay)
    pixel = lambda xyz: bev_pixels(np.asarray(xyz).reshape(-1, 3), bounds, size)
    scale = (size - 1) / (bounds[2] - bounds[0])
    meta = dict(
        format="lingbot-navi-offline-render-v1",
        asset_file=asset_path.name,
        gaussian_count=len(renderer.xyz),
        sh_degree=renderer.sh_degree,
        source_scene=scene.get("source_scene"),
        coordinate_frame="asset_world_meters_z_up",
        asset_from_first_block=transform.tolist(),
        model_result="result.json",
        perspective_renderer=PERSPECTIVE_RENDERER,
        perspective_depth="expected radial distance in meters",
        perspective_background="black",
        bev=dict(
            z_min=z_min,
            z_max=z_max,
            bounds_xy=bounds.tolist(),
            size=size,
            meters_per_pixel=1 / scale,
            annotated_image_offset_xy=[0, 58],
            camera_to_asset_cv=bev_c2w.tolist(),
            K=bev_k.tolist(),
            clip_rule="retain Gaussian centers within the Z interval",
            selected_gaussians=int(selected.sum()),
            axes="+X right, +Y up",
            pixel_formula="u=(x-xmin)*scale; v=(ymax-y)*scale",
        ),
        finish_reason=result.get("finish_reason"),
        objects=[],
        poses=[],
        trajectory_asset_m=route.tolist(),
        goal_asset_m=goal.tolist() if goal is not None else None,
        notes=[
            "BEV is a cutaway render, not an occupancy or traversability map.",
            "Novel views can contain holes; the original Gaussian PLY retains view-dependent SH.",
            "Predictions are visualized unchanged, including incomplete or inaccurate results.",
        ],
    )
    lines = [
        f"Asset frame: +X right / +Y up / meters | Z cut: [{z_min:.2f}, {z_max:.2f}]",
        f"Finish: {result.get('finish_reason', 'unknown')} | "
        + ("Blue: current position  Green: frontier" if frontier is not None
           else "Red: target  Blue: camera  Green: route"),
    ]
    if exploration:
        c = pixel([current])[0]
        dot(draw, c, BLUE)
        label(draw, c + [10, 10], "Current position", BLUE)
        lines.append(f"Exploration: {exploration['status']} | current position supplied in first-block frame")
        if frontier is not None:
            f = pixel([frontier])[0]
            dot(draw, f, GREEN, radius=10)
            label(draw, f + [12, -25], "Predicted frontier", GREEN)
            lines.append("Frontier prediction only; the full-scene render does not show memory coverage.")
        from textwrap import wrap
        for message in exploration["messages"]:
            content = message["content"]
            if isinstance(content, list):
                content = "\n".join(part.get("text", "[Target image]") for part in content)
            text = message["role"].capitalize() + ": " + content.replace("<|im_end|>", "")
            for paragraph in text.splitlines():
                lines.extend(wrap(paragraph, width=110) or [""])
        meta["exploration"] = dict(current_asset_xyz=current.tolist(),
            frontier_asset_xyz=frontier.tolist() if frontier is not None else None,
            status=exploration["status"])
    if len(route):
        poly = [tuple(x) for x in pixel(route)]
        draw.line(poly, fill="white", width=8)
        draw.line(poly, fill=GREEN, width=4)
        for j in np.linspace(0, len(poly) - 2, min(5, len(poly) - 1), dtype=int):
            arrow(draw, poly[j], poly[j + 1], GREEN, 3)
        dot(draw, poly[0], BLUE)
        dot(draw, poly[-1], GREEN)
        label(draw, np.asarray(poly[0]) + [10, 10], "Start", BLUE)
        label(draw, np.asarray(poly[-1]) + [10, 10], "End", GREEN)
        lines.append(
            f"Predicted trajectory: {len(route)} points; displayed without smoothing or replanning."
        )
    if goal is not None:
        g = pixel(goal)[0]
        draw.rectangle((g[0] - 9, g[1] - 9, g[0] + 9, g[1] + 9), outline=GREEN, width=3)
        label(draw, g + [10, -25], "Goal", GREEN)
    # Perspective renders use the full asset, not the BEV cutaway.
    width, height = 800, 600
    k = np.array([[400.0, 0.0, width / 2], [0.0, 400.0, height / 2], [0.0, 0.0, 1.0]])
    for i, target in enumerate(world_centers):
        chosen, candidates = observation_candidates(renderer, views, target)
        camera = np.array(chosen["camera_to_asset_cv"])
        rgb = renderer.render(camera, k, width, height).rgb
        raw_name, name = f"object_{i:02}_rgb.png", f"object_{i:02}.png"
        rgb.save(output_dir / raw_name)
        marker, depth = pinhole_pixels(target[None], camera, k)
        assert depth[0] > 0
        detail = rgb.copy()
        dd = ImageDraw.Draw(detail)
        x, y = marker[0]
        dd.ellipse((x - 16, y - 16, x + 16, y + 16), outline=RED, width=3)
        dd.line((x - 24, y, x + 24, y), fill=RED, width=2)
        dd.line((x, y - 24, x, y + 24), fill=RED, width=2)
        visible = chosen["likely_visible"]
        status = (
            "likely visible"
            if visible
            else "occluded / unsupported: marker is projected"
        )
        card(
            detail,
            f"Object {i + 1} / selected observation",
            [
                f"Saved position {chosen['saved_view_index'] + 1}, aimed at target | {status}",
                "Model XYZ: " + ", ".join(f"{x:+.3f}" for x in centers[i]) + " m",
                "Asset XYZ: " + ", ".join(f"{x:+.3f}" for x in target) + " m",
            ],
        ).save(output_dir / name)
        t = pixel(target)[0]
        c = pixel(camera[:3, 3])[0]
        draw.line([tuple(c), tuple(t)], fill=BLUE, width=2)
        dot(draw, c, BLUE, 5)
        label(draw, c + [9, 8], f"View {i + 1}", BLUE, 16)
        dot(draw, t, RED)
        label(draw, t + [10, -27], f"Object {i + 1}", RED)
        inside = bool(z_min <= target[2] <= z_max)
        in_bev = bool(
            bounds[0] <= target[0] <= bounds[2] and bounds[1] <= target[1] <= bounds[3]
        )
        lines.append(
            f"Object {i + 1}: asset XYZ ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f})"
            + (" | outside Z slice" if not inside else "")
            + (" | outside BEV bounds" if not in_bev else "")
        )
        meta["objects"].append(
            dict(
                index=i,
                model_xyz=centers[i].tolist(),
                asset_xyz=target.tolist(),
                inside_z_slice=inside,
                inside_bev_bounds=in_bev,
                selected_view=chosen,
                candidates=candidates,
                K=k.tolist(),
                width=width,
                height=height,
                image=name,
                raw_image=raw_name,
                selection_rule="prefer likely-visible, then 0.75--6m, then foreground fraction, then distance to 2.5m",
            )
        )
    for i, camera in enumerate(cameras):
        c = camera[:3, 3]
        forward = camera[:3, 2]
        pp = pixel([c, c + forward * 0.8])
        dot(draw, pp[0], BLUE)
        arrow(draw, pp[0], pp[1], BLUE)
        label(
            draw,
            pp[0] + [10, 10],
            f"Start / pose {i + 1}" if len(route) and i == 0 else f"Pose {i + 1}",
            BLUE,
        )
        info = dict(
            index=i,
            model_pose_xyzw=poses[i].tolist(),
            camera_to_asset_cv=camera.tolist(),
        )
        # Navigation stays on BEV; do not produce a redundant localization report.
        if not len(route) and goal is None:
            view = next(
                (
                    v
                    for v in views
                    if image is not None
                    and v.get("image")
                    and (scene_path.parent / v["image"]).resolve()
                    == Path(image).expanduser().resolve()
                ),
                views[0] if views else None,
            )
            pw, ph = (view["width"], view["height"]) if view else (640, 480)
            pk = (
                np.array(view["K"])
                if view
                else np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
            )
            observation = renderer.render(camera, pk, pw, ph)
            rgb, alpha = observation.rgb, observation.alpha
            name = f"pose_{i:02}_rgb.png"
            rgb.save(output_dir / name)
            info.update(
                image=name,
                K=pk.tolist(),
                width=pw,
                height=ph,
                render_coverage=float((alpha > 0.5).mean()),
                intrinsics_source="matching scene view or scene.views[0]"
                if view
                else "default 90-degree horizontal FOV",
            )
            panels = [rgb]
            titles = ["Render at predicted pose"]
            if input_image is not None:
                panels.insert(0, ImageOps.contain(input_image, (pw, ph)))
                titles.insert(0, "Input image")
            combined = Image.new("RGB", (pw * len(panels), ph + 40), "#f5f6f3")
            cd = ImageDraw.Draw(combined)
            for j, panel in enumerate(panels):
                combined.paste(
                    panel,
                    (j * pw + (pw - panel.width) // 2, 40 + (ph - panel.height) // 2),
                )
                cd.text((j * pw + 15, 10), titles[j], fill=INK, font=font(19))
            comparison = f"pose_{i:02}.png"
            card(
                combined,
                f"Image localization / pose {i + 1}",
                [
                    "Asset position: " + ", ".join(f"{x:+.3f}" for x in c) + " m",
                    "Rendered from the predicted pose; no alignment correction.",
                ],
            ).save(output_dir / comparison)
            info["comparison"] = comparison
        meta["poses"].append(info)
        lines.append(f"Pose {i + 1}: asset XYZ ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})")
    # A scale bar and axis compass do not depend on scene orientation.
    length = min(1.0, (bounds[2] - bounds[0]) * 0.2)
    draw.line([(28, size - 34), (28 + length * scale, size - 34)], fill=INK, width=4)
    label(draw, (28, size - 65), f"{length:g} m", size=16)
    arrow(draw, (size - 85, size - 40), (size - 30, size - 40), INK, 2)
    arrow(draw, (size - 85, size - 40), (size - 85, size - 95), INK, 2)
    label(draw, (size - 27, size - 49), "X", size=15)
    label(draw, (size - 90, size - 118), "Y", size=15)
    card(overlay, "LingBot-Navi / BEV cutaway", lines).save(output_dir / "bev.png")
    (output_dir / "render.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    return meta
