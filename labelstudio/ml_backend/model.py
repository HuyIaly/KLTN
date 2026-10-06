"""Label Studio ML backend (v2) cho YOLO layout CV.

Label Studio gọi /predict khi annotator mở task (hoặc khi bấm "Retrieve Predictions"),
backend trả về box gợi ý; annotator chỉ cần sửa thay vì vẽ từ đầu.
Hỗ trợ cả template Multi-page (valueList, có item_index) lẫn ảnh đơn.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from label_studio_ml.model import LabelStudioMLBase  # noqa: E402
from label_studio_ml.response import ModelResponse  # noqa: E402

from config import BASE_DIR, IMGSZ, active_weights, auto_device  # noqa: E402
from labelstudio.core import (CACHE, detect, open_rgb, parse_label_config,  # noqa: E402
                              resolve_local, task_score, to_ls_results)

log = logging.getLogger(__name__)

# Mặc định lấy từ config.py; env chỉ để ghi đè khi cần.
# Trọng số được chọn lại MỖI request (config.active_weights): YOLO_WEIGHTS > finetune.pt > BEST_WEIGHTS
CONF = float(os.getenv("YOLO_CONF", "0.25"))
IOU = float(os.getenv("YOLO_IOU", "0.5"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", str(IMGSZ)))
DEVICE = os.getenv("YOLO_DEVICE") or auto_device()
DOC_ROOT = os.getenv("LOCAL_FILES_DOCUMENT_ROOT", str(BASE_DIR))  # = document root của Label Studio
LABEL_MAP = json.loads(os.getenv("LABEL_MAP", "{}"))      # {"ten_lop_yolo": "ten_label_LS"}


class CVLayoutBackend(LabelStudioMLBase):

    def setup(self):
        w = str(active_weights())
        if Path(w).exists():
            self.set("model_version", CACHE.version(w))

    def _image_path(self, url: str, task_id) -> str:
        p = resolve_local(url, DOC_ROOT)          # nhanh: đọc thẳng từ ổ đĩa
        if p is not None:
            return str(p)
        return self.get_local_path(url, task_id=task_id)  # fallback: tải qua API Label Studio

    def predict(self, tasks, context=None, **kwargs) -> ModelResponse:
        cfg = parse_label_config(self.label_config)
        weights = str(active_weights())
        model = CACHE.get(weights)
        version = CACHE.version(weights)

        missing = sorted({str(n) for n in model.names.values()} - set(LABEL_MAP) - set(cfg.labels))
        if missing:
            log.warning("Lớp YOLO không có trong label config (sẽ bị bỏ): %s", missing)

        predictions = []
        for task in tasks:
            urls = task["data"].get(cfg.value_key)
            urls = urls if isinstance(urls, list) else [urls]
            results, scores = [], []
            for k, url in enumerate(urls):
                try:
                    img = open_rgb(self._image_path(url, task.get("id")))
                except Exception as e:  # noqa: BLE001
                    log.error("Task %s trang %s: không đọc được ảnh %s (%s)", task.get("id"), k, url, e)
                    continue
                dets = detect(model, img, CONF, IOU, IMGSZ, DEVICE)
                r, s = to_ls_results(dets, img.width, img.height, cfg,
                                     item_index=k if cfg.multipage else None, label_map=LABEL_MAP)
                results += r
                scores += s
            predictions.append({"result": results, "score": task_score(scores), "model_version": version})
        return ModelResponse(predictions=predictions, model_version=version)

    def fit(self, event, data, **kwargs):
        # Không train YOLO trong webhook (quá nặng, dễ timeout). Retrain theo lô bằng
        # labelstudio/retrain.py; backend tự nạp lại khi weights đổi.
        log.info("Nhận sự kiện %s – retrain chạy riêng bằng labelstudio/retrain.py", event)
