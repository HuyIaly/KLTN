"""OCR cho trang không có text layer. Mỗi engine trả về List[Word] với bbox pixel."""
from __future__ import annotations

import shutil
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from PIL import Image

from .schema import Word


class BaseOCR:
    name = "base"

    def read(self, image: np.ndarray) -> list[Word]:
        raise NotImplementedError


class EasyOCREngine(BaseOCR):
    """Cài bằng pip, có tiếng Việt ('vi'). Trả về cụm chữ theo dòng/đoạn ngắn."""
    name = "easyocr"

    def __init__(self, langs: Sequence[str] = ("vi", "en"), gpu: bool | None = None):
        import easyocr

        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> list[Word]:
        out: list[Word] = []
        for pts, text, conf in self.reader.readtext(image, paragraph=False):
            text = text.strip()
            if not text:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            out.append(Word(text, (min(xs), min(ys), max(xs), max(ys)), float(conf)))
        return out


class TesseractEngine(BaseOCR):
    """Cần cài Tesseract + traineddata 'vie'. Trả về từng từ."""
    name = "tesseract"
    _WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def __init__(self, lang: str = "vie+eng", cmd: str | None = None, psm: int = 3):
        import pytesseract

        cmd = cmd or shutil.which("tesseract") or (self._WIN_PATH if Path(self._WIN_PATH).exists() else None)
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._pt = pytesseract
        self.lang, self.psm = lang, psm

    def read(self, image: np.ndarray) -> list[Word]:
        d = self._pt.image_to_data(Image.fromarray(image), lang=self.lang,
                                   config=f"--psm {self.psm}", output_type=self._pt.Output.DICT)
        out: list[Word] = []
        for i, text in enumerate(d["text"]):
            text = (text or "").strip()
            conf = float(d["conf"][i])
            if not text or conf < 0:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append(Word(text, (float(x), float(y), float(x + w), float(y + h)), conf / 100.0))
        return out


def create_ocr(engine: str = "easyocr", langs: Sequence[str] = ("vi", "en"),
               tesseract_cmd: str | None = None) -> BaseOCR:
    engine = engine.lower()
    if engine == "easyocr":
        return EasyOCREngine(langs)
    if engine == "tesseract":
        tess = {"vi": "vie", "en": "eng"}
        return TesseractEngine("+".join(tess.get(lang, lang) for lang in langs), cmd=tesseract_cmd)
    raise ValueError(f"OCR engine không hỗ trợ: {engine}")
