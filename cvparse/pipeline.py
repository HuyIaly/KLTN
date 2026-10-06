"""Ghép toàn bộ pipeline: CV -> chuẩn hoá -> YOLO -> text layer / OCR -> hậu xử lý -> dict/JSON.

Đầu ra dừng ở bước Hậu xử lý; trường `sections` là đầu vào trực tiếp cho NER chạy theo từng vùng.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Sequence

from PIL import Image

from .extract import assign_words, ocr_crop, text_layer_words
from .geometry import union
from .layout import LayoutDetector
from .normalize import NormalizedDocument, load_document
from .ocr import BaseOCR, create_ocr
from .postprocess import (build_lines, merge_sections, reading_order, snap_bbox,
                          strip_headers_footers)
from .schema import UNASSIGNED, Page, Region, Section, Word
from .visualize import draw_regions


@dataclass
class PipelineConfig:
    dpi: int = 200
    max_pages: int = 10
    conf: float = 0.25
    iou: float = 0.5
    text_mode: str = "auto"            # auto | text_layer | ocr
    ocr_fallback_empty: bool = True    # trang có text layer nhưng vùng rỗng -> OCR crop
    skip_ocr_labels: tuple[str, ...] = ("PHOTO", "AVATAR", "IMAGE", "LOGO")
    min_ioa: float = 0.5
    snap_boxes: bool = True
    snap_pad: float = 4.0
    order_strategy: str = "x_first"    # x_first | y_first
    remove_header_footer: bool = True
    merge_same_page: bool = True
    merge_cross_page: bool = True
    keep_unassigned: bool = True


@dataclass
class PipelineResult:
    source: str
    weights: str
    pages: list[Page]
    regions: list[Region]             # theo thứ tự đọc, UNASSIGNED ở cuối mỗi trang
    sections: list[Section]
    config: PipelineConfig
    timings: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ export
    def to_dict(self, include_lines: bool = True) -> dict:
        def reg(r: Region) -> dict:
            d = {"order": r.order, "page": r.page, "label": r.label, "score": round(r.score, 4),
                 "bbox": [round(v, 1) for v in r.bbox], "det_bbox": [round(v, 1) for v in r.det_bbox],
                 "source": r.source, "text": r.text}
            if include_lines:
                d["lines"] = [{"text": line.text, "bbox": [round(v, 1) for v in line.bbox],
                               "conf": round(line.conf, 3)} for line in r.lines]
            return d

        return {
            "source": self.source,
            "model": self.weights,
            "num_pages": len(self.pages),
            "pages": [{"index": p.index, "width": p.width, "height": p.height,
                       "has_text_layer": p.has_text_layer} for p in self.pages],
            "sections": [{"id": s.id, "label": s.label, "pages": s.pages, "score": round(s.score, 4),
                          "source": s.source, "region_orders": [r.order for r in s.parts],
                          "text": s.text} for s in self.sections],
            "regions": [reg(r) for r in self.regions],
            "timings_sec": {k: round(v, 3) for k, v in self.timings.items()},
            "config": asdict(self.config),
        }

    def to_markdown(self) -> str:
        out = []
        for s in self.sections:
            pages = ", ".join(str(p + 1) for p in s.pages)
            out.append(f"### {s.id}. {s.label}  _(trang {pages}, score {s.score:.2f}, {s.source})_\n")
            out.append(s.text.replace("\n", "  \n") if s.text else "_(không có chữ)_")
            out.append("")
        un = [r for r in self.regions if r.label == UNASSIGNED and r.lines]
        if un:
            out.append("### ⚠️ Chữ không thuộc vùng nào")
            for r in un:
                out.append(f"_Trang {r.page + 1}:_  \n" + r.text.replace("\n", "  \n"))
        return "\n".join(out)

    def annotated_images(self) -> list[Image.Image]:
        return [draw_regions(p.image, [r for r in self.regions if r.page == p.index]) for p in self.pages]


class CVLayoutPipeline:
    def __init__(self, weights, device: str | None = None, imgsz: int = 1024,
                 ocr_engine: str = "easyocr", ocr_langs: Sequence[str] = ("vi", "en"),
                 tesseract_cmd: str | None = None, detector: LayoutDetector | None = None):
        self.detector = detector or LayoutDetector(weights, device=device, imgsz=imgsz)
        self.weights = str(weights)
        self.ocr_engine, self.ocr_langs, self.tesseract_cmd = ocr_engine, tuple(ocr_langs), tesseract_cmd
        self._ocr: BaseOCR | None = None

    @property
    def ocr(self) -> BaseOCR:
        if self._ocr is None:  # khởi tạo trễ: PDF có text layer sẽ không phải load model OCR
            self._ocr = create_ocr(self.ocr_engine, self.ocr_langs, self.tesseract_cmd)
        return self._ocr

    # ------------------------------------------------------------------ steps
    def _page_words(self, doc: NormalizedDocument, page: Page, cfg: PipelineConfig) -> tuple[list[Word], str]:
        if cfg.text_mode != "ocr" and page.has_text_layer and doc.pdf is not None:
            return text_layer_words(doc.pdf[page.index], page.zoom), "text_layer"
        if cfg.text_mode == "text_layer":
            return [], "none"
        return self.ocr.read(page.image), "ocr"

    def run(self, path, cfg: PipelineConfig | None = None) -> PipelineResult:
        cfg = cfg or PipelineConfig()
        timings = {"normalize": 0.0, "layout": 0.0, "extract": 0.0, "postprocess": 0.0}

        t = time.perf_counter()
        doc = load_document(path, dpi=cfg.dpi, max_pages=cfg.max_pages)
        timings["normalize"] = time.perf_counter() - t

        ordered: list[Region] = []
        try:
            for page in doc.pages:
                t = time.perf_counter()
                regions = self.detector.detect(page.image, conf=cfg.conf, iou=cfg.iou, page_index=page.index)
                timings["layout"] += time.perf_counter() - t

                t = time.perf_counter()
                words, source = self._page_words(doc, page, cfg)
                leftovers = assign_words(regions, words, cfg.min_ioa)
                for r in regions:
                    r.source = source
                if source == "text_layer" and cfg.ocr_fallback_empty and cfg.text_mode == "auto":
                    skip = {s.upper() for s in cfg.skip_ocr_labels}
                    for r in regions:
                        if not r.words and r.label.upper() not in skip:
                            r.words = ocr_crop(self.ocr, page.image, r.bbox)
                            if r.words:
                                r.source = "ocr_fallback"
                timings["extract"] += time.perf_counter() - t

                t = time.perf_counter()
                for r in regions:
                    r.lines = build_lines(r.words)
                    if cfg.snap_boxes:
                        r.bbox = snap_bbox(r, cfg.snap_pad, page.width, page.height)
                order = reading_order([r.det_bbox for r in regions], page.width, page.height,
                                      strategy=cfg.order_strategy)
                ordered.extend(regions[i] for i in order)
                if leftovers and cfg.keep_unassigned:
                    u = Region(page=page.index, label=UNASSIGNED, bbox=union(w.bbox for w in leftovers),
                               score=0.0, words=leftovers, source=source)
                    u.lines = build_lines(leftovers)
                    ordered.append(u)
                timings["postprocess"] += time.perf_counter() - t

            t = time.perf_counter()
            if cfg.remove_header_footer:
                strip_headers_footers(ordered, {p.index: p.height for p in doc.pages})
            ordered = [r for r in ordered if r.label != UNASSIGNED or r.lines]
            for i, r in enumerate(ordered):
                r.order = i
            sections = merge_sections([r for r in ordered if r.label != UNASSIGNED],
                                      cfg.merge_same_page, cfg.merge_cross_page)
            timings["postprocess"] += time.perf_counter() - t
            pages = doc.pages
        finally:
            doc.close()

        timings["total"] = sum(timings.values())
        return PipelineResult(source=str(Path(path).name), weights=self.weights, pages=pages,
                              regions=ordered, sections=sections, config=cfg, timings=timings)
