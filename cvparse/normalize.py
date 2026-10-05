"""Bước 1 – Chuẩn hoá đầu vào: DOCX -> PDF -> ảnh trang; ảnh -> danh sách trang RGB.

PDF được giữ lại (không đóng) để bước sau lấy text layer đúng toạ độ.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps, ImageSequence

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz

from .schema import Page

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OFFICE_EXTS = {".docx", ".doc", ".odt", ".rtf"}

_SOFFICE_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
]


@dataclass
class NormalizedDocument:
    source: Path
    kind: str                       # "pdf" | "image"
    pages: List[Page]
    pdf: Optional["fitz.Document"] = None

    def close(self) -> None:
        if self.pdf is not None:
            self.pdf.close()
            self.pdf = None


# ----------------------------------------------------------------------------- DOCX -> PDF
def _find_soffice() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        p = shutil.which(name)
        if p:
            return p
    for p in _SOFFICE_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def office_to_pdf(path: Path, out_dir: Path, timeout: int = 180) -> Path:
    """Ưu tiên LibreOffice (headless); trên Windows có thể fallback sang docx2pdf (cần MS Word)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{path.stem}.pdf"

    soffice = _find_soffice()
    if soffice:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
            check=True, timeout=timeout, capture_output=True,
        )
        if target.exists():
            return target

    if path.suffix.lower() == ".docx":
        try:
            from docx2pdf import convert  # type: ignore
            convert(str(path), str(target))
            if target.exists():
                return target
        except ImportError:
            pass

    raise RuntimeError(
        "Không chuyển được DOCX sang PDF. Cài LibreOffice (khuyến nghị) hoặc `pip install docx2pdf` "
        "(Windows + MS Word)."
    )


# ----------------------------------------------------------------------------- text layer
def page_has_text_layer(page: "fitz.Page", min_chars: int = 30, max_bad_ratio: float = 0.1) -> bool:
    """PDF 'thật' có text layer dùng được. PDF scan / font hỏng (ký tự \ufffd) -> False => OCR."""
    chars = "".join(page.get_text("text").split())
    if len(chars) < min_chars:
        return False
    bad = sum(1 for c in chars if c == "\ufffd" or ord(c) < 32)
    return bad / len(chars) < max_bad_ratio


def _pixmap_to_rgb(pix: "fitz.Pixmap") -> np.ndarray:
    buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return buf[:, : pix.width * 3].reshape(pix.height, pix.width, 3).copy()


def _load_pdf(pdf_path: Path, dpi: int, max_pages: int, source: Path) -> NormalizedDocument:
    doc = fitz.open(str(pdf_path))
    if doc.needs_pass:
        doc.close()
        raise ValueError(f"PDF có mật khẩu: {source}")
    zoom = dpi / 72.0
    pages: List[Page] = []
    for i in range(min(len(doc), max_pages)):
        pg = doc[i]
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        pages.append(Page(index=i, image=_pixmap_to_rgb(pix), zoom=zoom,
                          has_text_layer=page_has_text_layer(pg)))
    return NormalizedDocument(source=source, kind="pdf", pages=pages, pdf=doc)


def _load_image(path: Path, max_pages: int) -> NormalizedDocument:
    pages: List[Page] = []
    with Image.open(path) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):  # TIFF nhiều trang
            if i >= max_pages:
                break
            frame = ImageOps.exif_transpose(frame.copy()).convert("RGB")
            pages.append(Page(index=i, image=np.asarray(frame).copy(), zoom=None, has_text_layer=False))
    return NormalizedDocument(source=path, kind="image", pages=pages, pdf=None)


def load_document(path, dpi: int = 200, max_pages: int = 10,
                  work_dir: Optional[Path] = None) -> NormalizedDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in OFFICE_EXTS:
        work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cvparse_"))
        return _load_pdf(office_to_pdf(path, work_dir), dpi, max_pages, source=path)
    if ext == ".pdf":
        return _load_pdf(path, dpi, max_pages, source=path)
    if ext in IMAGE_EXTS:
        return _load_image(path, max_pages)
    raise ValueError(f"Định dạng không hỗ trợ: {ext}")
