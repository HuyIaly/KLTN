"""Các kiểu dữ liệu dùng chung. Mọi bbox là (x0, y0, x1, y1) theo PIXEL trên ảnh trang."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]
UNASSIGNED = "UNASSIGNED"


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float = 1.0


@dataclass
class Line:
    text: str
    bbox: BBox
    conf: float = 1.0


@dataclass
class Page:
    index: int
    image: np.ndarray            # RGB uint8 (H, W, 3)
    zoom: Optional[float] = None  # pixel / point (chỉ có với PDF)
    has_text_layer: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class Region:
    page: int
    label: str
    bbox: BBox                    # bbox sau khi "snap" theo chữ (nếu bật)
    score: float
    det_bbox: Optional[BBox] = None  # bbox gốc YOLO trả về
    words: List[Word] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    source: str = ""              # text_layer | ocr | ocr_fallback | none
    order: int = -1

    def __post_init__(self):
        if self.det_bbox is None:
            self.det_bbox = tuple(self.bbox)

    @property
    def text(self) -> str:
        from .postprocess import join_lines
        return join_lines([l.text for l in self.lines])


@dataclass
class Section:
    """Một mục logic sau khi nối các vùng cùng nhãn (có thể trải qua nhiều trang)."""
    id: int
    label: str
    parts: List[Region]

    @property
    def lines(self) -> List[Line]:
        return [l for r in self.parts for l in r.lines]

    @property
    def text(self) -> str:
        from .postprocess import join_lines
        return join_lines([l.text for l in self.lines])

    @property
    def pages(self) -> List[int]:
        return sorted({r.page for r in self.parts})

    @property
    def score(self) -> float:
        return float(max(r.score for r in self.parts))

    @property
    def source(self) -> str:
        return ",".join(sorted({r.source for r in self.parts if r.source}))
