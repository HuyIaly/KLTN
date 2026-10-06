from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import UNASSIGNED, Region

_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
            (70, 160, 160), (240, 50, 230), (110, 140, 20), (0, 128, 128), (170, 110, 40),
            (128, 0, 0), (0, 0, 128), (128, 128, 0), (200, 80, 120)]


def color_for(label: str):
    if label == UNASSIGNED:
        return (128, 128, 128)
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


@lru_cache(maxsize=8)
def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_regions(image: np.ndarray, regions: list[Region]) -> Image.Image:
    im = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    font = _font(max(14, im.width // 75))
    lw = max(2, im.width // 600)
    for r in regions:
        col = color_for(r.label)
        x0, y0, x1, y1 = r.bbox
        draw.rectangle([x0, y0, x1, y1], outline=col + (255,), width=lw, fill=col + (28,))
        tag = UNASSIGNED if r.label == UNASSIGNED else f"#{r.order} {r.label} {r.score:.2f}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y0 - th - 6 if y0 - th - 6 >= 0 else y0
        draw.rectangle([x0, ty, x0 + tw + 8, ty + th + 6], fill=col + (225,))
        draw.text((x0 + 4, ty + 2 - tb[1]), tag, fill=(255, 255, 255), font=font)
    return im
