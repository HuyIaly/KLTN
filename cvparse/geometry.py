from __future__ import annotations

from typing import Iterable

from .schema import BBox


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x1 - x0) * (y1 - y0) if x1 > x0 and y1 > y0 else 0.0


def ioa(a: BBox, b: BBox) -> float:
    """Intersection over area của a (tỉ lệ a nằm trong b)."""
    ar = area(a)
    return intersection(a, b) / ar if ar > 0 else 0.0


def union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def expand(b: BBox, pad: float, w: float, h: float) -> BBox:
    return (max(0.0, b[0] - pad), max(0.0, b[1] - pad),
            min(float(w), b[2] + pad), min(float(h), b[3] + pad))
