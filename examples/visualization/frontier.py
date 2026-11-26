"""Show a retained capture path and a frontier prediction on the final OCC coverage."""
import textwrap

import numpy as np
from PIL import Image, ImageDraw
from data_toolkit.render import GaussianScene
from data_toolkit.render.geometry import bev_camera, bev_pixels, transform_points
from .drawing import font


def coverage_map(bev, bounds, archive, scene_id, retained):
    from scipy.ndimage import binary_dilation
    with np.load(archive, allow_pickle=False) as data:
        ids = data["block_ids"].tolist()
        if str(data["scene_id"]) != scene_id or not retained or not set(retained) <= set(ids):
            raise ValueError("Coverage must match the scene and retained memory blocks")
        known = data["block_coverage"][[ids.index(name) for name in retained]].any(axis=0)
        free = data["state"] == 1
        origin = data["bounds"][:2]
        cell = float(data["cell"])
    edge = known & binary_dilation(free & ~known)
    x = np.floor((np.linspace(bounds[0], bounds[2], bev.width) - origin[0]) / cell).astype(int)
    y = np.floor((np.linspace(bounds[3], bounds[1], bev.height) - origin[1]) / cell).astype(int)
    valid = (x >= 0) & (x < known.shape[1]) & ((y >= 0) & (y < known.shape[0]))[:, None]
    indices = np.ix_(y.clip(0, known.shape[0]-1), x.clip(0, known.shape[1]-1))
    rgb = np.array(bev)
    dark = ~(valid & known[indices])
    rgb[dark] = rgb[dark] * .22 + np.array([6, 13, 16]) * .78
    rgb[valid & edge[indices]] = [232, 199, 134]
    rows, cols = np.nonzero(edge)
    centers = origin + (np.column_stack((cols, rows)) + .5) * cell
    return Image.fromarray(rgb), centers


def render_search(out, question, result, cameras, capture, transform, settings, asset, device, coverage):
    renderer = GaussianScene(asset, device)
    size = 756
    floor = settings["floor_z"]
    pose, k, bounds = bev_camera(settings["bev_bounds"], floor + 4.6, size)
    bev = renderer.render(pose, k, size, size, ortho=True, z_range=(floor-.1, floor+1.6)).rgb
    base, centers = coverage_map(bev, bounds, coverage, capture["scene_id"], result["retained_blocks"])
    track = cameras[:, :3, 3]
    pixels = bev_pixels(track, bounds, size)
    frontier = transform_points(np.array(result["predicted_centers"]), transform)
    x, y = bev_pixels(frontier, bounds, size)[0]
    if len(centers):
        distances = np.linalg.norm(centers - frontier[0, :2], axis=1)
        nearest = distances.argmin()
        boundary = bev_pixels([[*centers[nearest], 0]], bounds, size)[0]
    title = question if question.lower().startswith("find ") else f"Find {question}"
    # Match the release figures' header, divider, type hierarchy and map key.
    width, height = 1680, 954
    margin, top = 48, 150
    sidebar = margin + size + 80
    frame = Image.new("RGB", (width, height), "#070e13")
    page = ImageDraw.Draw(frame)
    ink, muted, mint = "#edf2f3", "#9eafb6", "#98d8c9"
    page.text((margin, 24), "LingBot-Navi", font=font(22), fill=mint)
    page.text((margin-2, 55), "Frontier Exploration", font=font(54), fill=ink)
    block_name = " · ".join(name.replace("segment_", "Block ") for name in result["retained_blocks"])
    page.text((width-margin, 90), block_name, anchor="ra", font=font(26), fill=muted)
    page.line((margin, 125, width-margin, 125), fill="#334048", width=1)
    page.text((sidebar, 165), "Language goal", font=font(25), fill=muted)
    lines = textwrap.wrap(title, width=30)
    page.multiline_text((sidebar-2, 210), "\n".join(lines), font=font(46), fill=ink)
    y_key = 810
    page.line((sidebar, y_key-54, width-margin, y_key-54), fill="#334048", width=1)
    at = sidebar
    for index, (name, symbol) in enumerate([("Capture path", "line"), ("Current position", "ring"),
                         ("Open boundary", "cell"), ("Predicted frontier", "cross")]):
        at = sidebar + (index % 2) * 360
        y_key = 814 + (index // 2) * 55
        if symbol == "line":
            page.line((at, y_key, at+25, y_key), fill=mint, width=4)
        elif symbol == "ring":
            page.ellipse((at+6, y_key-6, at+18, y_key+6), outline=mint, width=3)
        elif symbol == "cell":
            page.rectangle((at+5, y_key-6, at+18, y_key+6), fill="#e8c786")
        else:
            page.line((at+5,y_key,at+19,y_key),fill=ink,width=3)
            page.line((at+12,y_key-7,at+12,y_key+7),fill=ink,width=3)
        page.text((at+34,y_key),name,font=font(25),anchor="lm",fill=muted)

    draw = ImageDraw.Draw(base)
    for i, point in enumerate(pixels):
        if i and np.linalg.norm(track[i] - track[i-1]) < .6:
            segment = [tuple(pixels[i-1]), tuple(point)]
            draw.line(segment, fill="#233f40", width=6)
            draw.line(segment, fill="#97d8cd", width=3)
    u, v = pixels[-1]
    draw.ellipse((u-5, v-5, u+5, v+5), fill="#97d8cd", outline="#183834", width=2)
    for color, stroke in [("#142020", 5), ("#ffffff", 3)]:
        draw.line((x-9, y, x+9, y), fill=color, width=stroke)
        draw.line((x, y-9, x, y+9), fill=color, width=stroke)
    if len(centers):
        detail_top = 372
        page.text((sidebar, detail_top-52), "Predicted frontier to nearest boundary cell",
                  font=font(25), fill=ink)
        w, h = width-margin-sidebar, 335
        zoom = min(2.7, w/(abs(boundary[0]-x)+60), h/(abs(boundary[1]-y)+60))
        center = (np.array([x,y]) + boundary) / 2
        left, upper = center - np.array([w,h]) / (2*zoom)
        detail = base.transform((w,h), Image.Transform.AFFINE,
                                (1/zoom,0,left,0,1/zoom,upper),
                                resample=Image.Resampling.BILINEAR, fillcolor="#070e13")
        dd = ImageDraw.Draw(detail)
        start = (np.array([x,y]) - [left,upper])*zoom
        end = (boundary - [left,upper])*zoom
        dd.line([tuple(start),tuple(end)],fill="#17232a",width=6)
        dd.line([tuple(start),tuple(end)],fill="#ffffff",width=2)
        for pos, color in [(start,"#ffffff"),(end,"#e8c786")]:
            u0,v0 = pos
            dd.ellipse((u0-5,v0-5,u0+5,v0+5),fill=color,outline="#162127",width=2)
        tx,ty = (start+end)/2
        label = f"{distances[nearest]:.2f} m"
        dd.rounded_rectangle((tx+14,ty-20,tx+120,ty+20),radius=5,fill="#0b171b")
        dd.text((tx+67,ty),label,anchor="mm",fill=ink,font=font(25))
        frame.paste(detail,(sidebar,detail_top))
    frame.paste(base, (margin, top))
    frame.save(out / "exploration.png")
