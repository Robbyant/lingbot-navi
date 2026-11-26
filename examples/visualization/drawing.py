"""Shared annotation styles for example images and videos."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RED = "#da4d42"
BLUE = "#2586c6"
GREEN = "#17846f"
INK = "#23332f"


def font(size):
    # Pillow's bundled font keeps rendering independent of OS font packages.
    return ImageFont.load_default(size=size)


def label(draw, position, text, color=INK, size=19):
    x, y = position
    box = draw.textbbox((x, y), text, font=font(size))
    draw.rounded_rectangle(
        (box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), radius=4, fill="#fffffff0"
    )
    draw.text((x, y), text, fill=color, font=font(size))


def dot(draw, xy, color, radius=7):
    x, y = xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline="white",
        width=2,
    )


def arrow(draw, a, b, color, width=4):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    v = b - a
    n = np.linalg.norm(v)
    if n > 1:
        v /= n
        side = np.array([-v[1], v[0]])
        draw.polygon(
            [tuple(b), tuple(b - 12 * v + 5 * side), tuple(b - 12 * v - 5 * side)],
            fill=color,
        )


def card(image, title, lines):
    out = Image.new(
        "RGB", (image.width, image.height + 76 + 27 * len(lines)), "#f5f6f3"
    )
    out.paste(image, (0, 58))
    draw = ImageDraw.Draw(out)
    draw.text((20, 18), title, fill=INK, font=font(24))
    for i, line in enumerate(lines):
        draw.text((20, image.height + 70 + i * 27), line, fill=INK, font=font(17))
    return out
