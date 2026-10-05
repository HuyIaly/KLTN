"""Bước 2 – Layout detection bằng YOLO (Ultralytics): ảnh trang -> các vùng theo mục CV."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from .geometry import area, ioa
from .schema import Region


class LayoutDetector:
    def __init__(self, weights, device: Optional[str] = None, imgsz: int = 1024):
        from ultralytics import YOLO  # import trễ để các module khác dùng được khi chưa cài

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.device = device
        self.imgsz = imgsz
        self.names = self.model.names  # {id: name}

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.5,
               page_index: int = 0, min_area_frac: float = 1e-4) -> List[Region]:
        h, w = image.shape[:2]
        res = self.model.predict(
            Image.fromarray(image),  # PIL -> tránh nhầm RGB/BGR
            conf=conf, iou=iou, imgsz=self.imgsz, device=self.device,
            agnostic_nms=True,        # 1 vùng chỉ nên có 1 nhãn mục
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        regions = []
        for b, c, s in zip(xyxy, cls, scores):
            box = tuple(float(v) for v in b)
            if area(box) < min_area_frac * w * h:
                continue
            regions.append(Region(page=page_index, label=str(self.names[int(c)]), bbox=box, score=float(s)))
        return remove_contained(regions)


def remove_contained(regions: List[Region], thr: float = 0.85) -> List[Region]:
    """Bỏ box cùng nhãn nằm gần trọn trong box khác (YOLO đôi khi ra box con trùng mục).
    Box khác nhãn lồng nhau (vd JOB_TITLE trong PROFILE) được giữ lại."""
    keep = []
    for i, r in enumerate(regions):
        dominated = any(
            j != i and o.label == r.label and area(o.bbox) >= area(r.bbox)
            and ioa(r.bbox, o.bbox) >= thr and (area(o.bbox) > area(r.bbox) or j < i)
            for j, o in enumerate(regions)
        )
        if not dominated:
            keep.append(r)
    return keep
