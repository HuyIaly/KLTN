"""Phần lõi dùng chung cho ML backend và script pre-annotate.

- Đọc label config của project (RectangleLabels + Image value / valueList)
- Nạp YOLO có cache + tự nạp lại khi file trọng số thay đổi (sau mỗi vòng retrain)
- Chuyển box YOLO (pixel) -> kết quả Label Studio (phần trăm, có item_index cho multi-page)
- Tìm file ảnh trên đĩa từ URL /data/local-files/?d=... (khỏi phải tải qua HTTP)
"""
from __future__ import annotations

import os
import re
import threading
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageOps

Detection = Tuple[str, float, float, float, float, float]  # label, x0, y0, x1, y1, score (pixel)


# ----------------------------------------------------------------------------- label config
@dataclass
class LSConfig:
    from_name: str             # tên tag RectangleLabels
    to_name: str               # tên tag Image
    value_key: str             # khoá trong task["data"]
    multipage: bool            # Image dùng valueList (1 task = nhiều trang)
    labels: List[str] = field(default_factory=list)


def parse_label_config(xml: str) -> LSConfig:
    root = ET.fromstring(xml)
    rect = next((e for e in root.iter() if e.tag == "RectangleLabels"), None)
    if rect is None:
        raise ValueError("Label config không có tag <RectangleLabels>")
    to_name = rect.get("toName", "").split(",")[0].strip()
    img = next((e for e in root.iter() if e.tag == "Image" and e.get("name") == to_name), None)
    if img is None:
        raise ValueError(f"Không tìm thấy <Image name='{to_name}'>")
    value_list, value = img.get("valueList"), img.get("value")
    return LSConfig(
        from_name=rect.get("name"), to_name=to_name,
        value_key=(value_list or value or "").lstrip("$"), multipage=bool(value_list),
        labels=[l.get("value") for l in rect.iter("Label") if l.get("value")],
    )


def build_label_config(classes: Sequence[str], multipage: bool = True,
                       from_name: str = "label", to_name: str = "page", value_key: str = "pages") -> str:
    """Sinh XML khớp với converter labelstudio_to_yolo.py."""
    img_attr = f'valueList="${value_key}"' if multipage else f'value="${value_key}"'
    labels = "\n".join(f'    <Label value="{c}"/>' for c in classes)
    return (f'<View>\n  <Image name="{to_name}" {img_attr} zoom="true" zoomControl="true"/>\n'
            f'  <RectangleLabels name="{from_name}" toName="{to_name}" strokeWidth="2">\n{labels}\n'
            f'  </RectangleLabels>\n</View>\n')


# ----------------------------------------------------------------------------- YOLO cache
class YoloCache:
    """ML backend v2 tạo instance model MỖI request -> giữ YOLO ở cấp module.
    Nếu file .pt đổi mtime (retrain xong ghi đè) thì tự nạp lại."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._key: Optional[Tuple[str, float]] = None

    def get(self, weights: str):
        mtime = os.path.getmtime(weights)
        with self._lock:
            if self._key != (weights, mtime):
                from ultralytics import YOLO
                self._model = YOLO(weights)
                self._key = (weights, mtime)
            return self._model

    @staticmethod
    def version(weights: str) -> str:
        ts = datetime.fromtimestamp(os.path.getmtime(weights)).strftime("%Y%m%d-%H%M")
        return f"{Path(weights).stem}@{ts}"


CACHE = YoloCache()


def detect(model, image: Image.Image, conf: float = 0.25, iou: float = 0.5,
           imgsz: int = 1024, device: Optional[str] = None) -> List[Detection]:
    res = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device,
                        agnostic_nms=True, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    names = model.names
    out = []
    for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                       res.boxes.conf.cpu().numpy()):
        out.append((str(names[int(c)]), *map(float, b), float(s)))
    return out


# ----------------------------------------------------------------------------- -> Label Studio
def to_ls_results(dets: List[Detection], width: int, height: int, cfg: LSConfig,
                  item_index: Optional[int] = None,
                  label_map: Optional[Dict[str, str]] = None) -> Tuple[List[dict], List[float]]:
    """Trả về (results, scores). Nhãn không có trong label config bị bỏ qua."""
    label_map = label_map or {}
    allowed = set(cfg.labels)
    results, scores = [], []
    for label, x0, y0, x1, y1, score in dets:
        label = label_map.get(label, label)
        if allowed and label not in allowed:
            continue
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        r = {
            "id": uuid.uuid4().hex[:10],
            "from_name": cfg.from_name, "to_name": cfg.to_name, "type": "rectanglelabels",
            "original_width": int(width), "original_height": int(height), "image_rotation": 0,
            "value": {"x": 100 * x0 / width, "y": 100 * y0 / height,
                      "width": 100 * (x1 - x0) / width, "height": 100 * (y1 - y0) / height,
                      "rotation": 0, "rectanglelabels": [label]},
            "score": round(score, 4),
        }
        if item_index is not None:
            r["item_index"] = int(item_index)
        results.append(r)
        scores.append(score)
    return results, scores


def task_score(scores: List[float]) -> float:
    """Điểm task = trung bình score box. LS sắp xếp task theo điểm này -> gán nhãn task
    model kém tự tin trước (active learning thủ công)."""
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ----------------------------------------------------------------------------- URL <-> file
def local_files_url(path: Path, doc_root: Path) -> str:
    rel = Path(path).resolve().relative_to(Path(doc_root).resolve())
    return "/data/local-files/?d=" + rel.as_posix()


def url_to_name(url: str) -> str:
    """Tên file ảnh trong URL task: local-files (?d=PNG/CV_1_page_001.png) hoặc upload
    (/data/upload/3/ab12cd34-CV_1_page_001.png -> bỏ prefix hash)."""
    u = urlparse(url)
    q = parse_qs(u.query)
    name = Path(unquote(q["d"][0] if "d" in q else u.path)).name
    m = re.match(r"^[0-9a-f]{6,}-(.+)$", name)
    return m.group(1) if m else name


def resolve_local(url: str, doc_root: Optional[str]) -> Optional[Path]:
    q = parse_qs(urlparse(url).query)
    if "d" in q and doc_root:
        p = Path(doc_root) / unquote(q["d"][0])
        if p.exists():
            return p
    p = Path(unquote(url))
    return p if p.exists() else None


def open_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        # trình duyệt hiển thị ảnh theo EXIF -> toạ độ % phải tính trên ảnh đã xoay theo EXIF
        return ImageOps.exif_transpose(im).convert("RGB")
