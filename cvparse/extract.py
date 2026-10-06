"""Bước 3 – Lấy chữ cho từng vùng.

- Có text layer : lấy word từ PDF (PyMuPDF), đổi point -> pixel theo zoom/rotation.
- Không có      : OCR CẢ TRANG một lần, rồi gán word vào vùng (nhất quán với nhánh PDF,
                  tránh OCR crop bị cắt chữ ở mép box YOLO).
Gán word -> vùng theo tâm/IoA, ưu tiên vùng nhỏ hơn khi lồng nhau (JOB_TITLE trong PROFILE).
"""
from __future__ import annotations


import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .geometry import area, expand, ioa
from .ocr import BaseOCR
from .schema import BBox, Region, Word


def text_layer_words(page: fitz.Page, zoom: float) -> list[Word]:
    # Toạ độ get_text là của trang CHƯA xoay; ảnh render là trang ĐÃ xoay -> nhân rotation_matrix.
    mat = page.rotation_matrix * fitz.Matrix(zoom, zoom)
    words: list[Word] = []
    for x0, y0, x1, y1, txt, *_ in page.get_text("words"):
        txt = txt.strip()
        if not txt:
            continue
        r = fitz.Rect(x0, y0, x1, y1) * mat
        words.append(Word(txt, (r.x0, r.y0, r.x1, r.y1), 1.0))
    return words


def assign_words(regions: list[Region], words: list[Word],
                 min_ioa: float = 0.5, tie_margin: float = 0.1) -> list[Word]:
    """Gán mỗi word cho đúng 1 vùng. Trả về các word không thuộc vùng nào."""
    leftovers: list[Word] = []
    for w in words:
        cands = []
        for i, r in enumerate(regions):
            v = ioa(w.bbox, r.bbox)
            if v >= min_ioa:
                cands.append((i, v, area(r.bbox)))
        if not cands:
            leftovers.append(w)
            continue
        best_v = max(c[1] for c in cands)
        i = min((c for c in cands if c[1] >= best_v - tie_margin), key=lambda c: c[2])[0]
        regions[i].words.append(w)
    return leftovers


def ocr_crop(engine: BaseOCR, image: np.ndarray, bbox: BBox, pad: int = 6) -> list[Word]:
    """OCR riêng một vùng (dùng khi trang có text layer nhưng vùng không có chữ, vd tiêu đề là ảnh)."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in expand(bbox, pad, w, h))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return []
    words = engine.read(np.ascontiguousarray(crop))
    return [Word(t.text, (t.bbox[0] + x0, t.bbox[1] + y0, t.bbox[2] + x0, t.bbox[3] + y0), t.conf)
            for t in words]
