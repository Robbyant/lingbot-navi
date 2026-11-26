"""Render and encode a local planner execution video."""

from contextlib import contextmanager
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from data_toolkit.render.geometry import bev_camera, bev_pixels
from .drawing import BLUE, GREEN, INK, arrow, dot, font


@contextmanager
def execution_video(renderer, camera, route, start, floor_z, output, *, ffmpeg, fps):
    """Yield a frame writer for synchronized first-person RGB and top-down paths."""
    xy = np.vstack((route[:, :2], start[:2]))
    bounds = [*(xy.min(0) - 1.0), *(xy.max(0) + 1.0)]
    map_size = 416
    bev_pose, bev_k, bounds = bev_camera(bounds, floor_z + 4.6, map_size)
    width = (camera.width + map_size + 1) // 2 * 2
    height = (max(camera.height, map_size) + 45) // 2 * 2
    map_base = renderer.render(
        bev_pose, bev_k, map_size, map_size, ortho=True,
        z_range=(floor_z - 0.1, floor_z + 1.6),
    ).rgb
    draw = ImageDraw.Draw(map_base)
    planned = [tuple(p) for p in bev_pixels(route, bounds, map_size)]
    draw.line(planned, fill="white", width=6)
    draw.line(planned, fill=GREEN, width=3)
    dot(draw, planned[-1], GREEN)
    command = [
        ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
        def write_frame(rgb, index, action_label, remaining, executed, forward):
            frame = Image.new("RGB", (width, height), "#f5f6f3")
            frame.paste(rgb, (0, 44))
            overlay = map_base.copy()
            draw = ImageDraw.Draw(overlay)
            trace = [tuple(p) for p in bev_pixels(executed, bounds, map_size)]
            if len(trace) > 1:
                draw.line(trace, fill=BLUE, width=4)
            position = executed[-1]
            here, facing = bev_pixels([position, position + forward * 0.5], bounds, map_size)
            arrow(draw, here, facing, BLUE)
            dot(draw, here, BLUE, radius=4)
            frame.paste(overlay, (camera.width, 44))
            draw = ImageDraw.Draw(frame)
            draw.text((12, 10), f"Step {index} | {action_label} | Route end {remaining:.2f} m", fill=INK, font=font(20))
            draw.text((camera.width + 12, 12), "Planned route", fill=GREEN, font=font(17))
            draw.text((camera.width + 128, 12), "Executed path", fill=BLUE, font=font(17))
            writer.stdin.write(frame.tobytes())

        try:
            yield write_frame
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg failed to encode the execution video")
        except BaseException:
            if writer.poll() is None:
                writer.kill()
            raise
