"""Draw capture trajectories and camera headings on a 3DGS top-down view."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from data_toolkit.render.geometry import bev_camera, bev_pixels
from .drawing import arrow, font


def render_trajectory_bev(renderer, trajectory, output_path):
    poses = np.asarray([f["camera_pose_c2w"] for f in trajectory["frames"]])
    settings = trajectory.get("visualization", {})
    floor = float(settings.get("floor_z", poses[0, 2, 3] - 1.25))
    bounds = settings.get(
        "bev_bounds", [*(poses[:, :2, 3].min(0) - 2), *(poses[:, :2, 3].max(0) + 2)]
    )
    # Include every capture position and heading tip even when the inference
    # demo's tighter room crop does not cover the entire capture trajectory.
    tips = poses[:, :3, 3] + poses[:, :3, 0] * 0.35
    extent = np.concatenate([poses[:, :2, 3], tips[:, :2]])
    bounds = [
        *np.minimum(bounds[:2], extent.min(0) - 0.4),
        *np.maximum(bounds[2:], extent.max(0) + 0.4),
    ]
    size = 900
    camera, k, bounds = bev_camera(bounds, floor + 4.6, size)
    base = renderer.render(
        camera, k, size, size, ortho=True, z_range=(floor - 0.1, floor + 1.6)
    ).rgb
    base.save(Path(output_path).with_name("trajectory_bev_base.png"))
    colors = ["#ee653b", "#2f9ce0", "#da58be", "#31b78a", "#f2bd32", "#8e79ea"]
    blocks = trajectory["blocks"]

    def overlay(indices):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for b in indices:
            segment = poses[blocks[b]["indices"]]
            pixels = bev_pixels(segment[:, :3, 3], bounds, size)
            color = colors[b % len(colors)]
            line = [tuple(v) for v in pixels]
            draw.line(line, fill="white", width=7)
            draw.line(line, fill=color, width=4)
            # Fixed spacing along the saved frame order also shows in-place turns.
            step = max(1, len(segment) // 12)
            for j in range(0, len(segment), step):
                tip = segment[j, :3, 3] + segment[j, :3, 0] * 0.35
                arrow(draw, pixels[j], bev_pixels(tip[None], bounds, size)[0], color)
            x, y = pixels[0]
            draw.ellipse(
                (x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2
            )
            x, y = pixels[-1]
            draw.rectangle(
                (x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2
            )
        return image

    bg = "#f5f6f3"
    # Overview and block legend.
    top = Image.new("RGB", (1800, 1040), bg)
    top.paste(overlay(range(len(blocks))), (30, 90))
    draw = ImageDraw.Draw(top)
    draw.text(
        (30, 22), "Spatial Memory capture trajectories", fill="#23332f", font=font(32)
    )
    draw.text(
        (980, 110),
        f"{len(blocks)} blocks / {len(poses)} capture frames",
        fill="#23332f",
        font=font(30),
    )
    for i, b in enumerate(blocks):
        draw.text(
            (980, 190 + i * 65),
            f"{b['name']}    {len(b['indices'])} frames",
            fill=colors[i % len(colors)],
            font=font(28),
        )
    for i, text in enumerate(
        [
            "Arrow: camera viewing direction",
            "Circle: first frame   Square: last frame",
            "Camera-to-world poses in the 3DGS asset frame",
            "BEV cutaway: floor - 0.1 m to floor + 1.6 m",
        ]
    ):
        draw.text((980, 650 + i * 52), text, fill="#23332f", font=font(23))
    top.save(output_path)
