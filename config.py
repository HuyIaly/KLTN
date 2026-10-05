"""
Cấu hình dùng chung cho toàn bộ pipeline CV layout.

Mọi script import từ đây, nên chỉ cần sửa đường dẫn / class ở MỘT chỗ.
Có thể đổi thư mục gốc mà không sửa code:  set CV_BASE_DIR=E:\\Other
"""

import os
from pathlib import Path

# ============================================================
# ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(os.environ.get("CV_BASE_DIR", r"D:\Final"))

PDF_DIR = BASE_DIR / "CV"                 # PDF gốc
PNG_DIR = BASE_DIR / "PNG"                # ảnh render từ PDF
DATASET_DIR = BASE_DIR / "YOLO_Dataset"   # dataset YOLO
RUNS_DIR = BASE_DIR / "runs"              # output training
RUN_NAME = "cv_layout_yolo"
BEST_WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

WEIGHTS_DIR = BASE_DIR / "weights"        # scratch.pt / finetune.pt (training/train.py copy vào đây)
SCRATCH_WEIGHTS = WEIGHTS_DIR / "scratch.pt"
FINETUNE_WEIGHTS = WEIGHTS_DIR / "finetune.pt"
LS_DIR = BASE_DIR / "labelstudio"         # export, tasks pre-annotate, label config
LS_ROUNDS_DIR = BASE_DIR / "ls_rounds"    # dữ liệu từng vòng retrain

# ============================================================
# LABEL STUDIO
# ============================================================

LS_PORT = 8080
ML_BACKEND_PORT = 9090
LS_URL = f"http://localhost:{LS_PORT}"
# Local storage của project trỏ vào PNG_DIR, document root = BASE_DIR
# -> URL ảnh: /data/local-files/?d=PNG/CV_1_page_001.png
PAGE_NAME = "{stem}_page_{page:03d}.png"   # tên ảnh trang khi render PDF

# ============================================================
# THAM SỐ ẢNH / MODEL
# ============================================================

RENDER_DPI = 300   # 300 DPI → trang A4 = 2482 × 3510 (đúng với dữ liệu gán nhãn)
IMGSZ = 1024       # PHẢI giống nhau giữa train / predict / demo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ============================================================
# CLASS
# ============================================================

CLASS_NAMES = [
    "NAME", "CONTACT", "SUMMARY", "EDUCATION", "EXPERIENCE", "SKILLS",
    "CERTIFICATION", "PROJECT", "ACHIEVEMENT", "LANGUAGE", "REFERENCE", "OTHER",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def class_name(class_id: int) -> str:
    return CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"CLASS_{class_id}"


def natural_key(path) -> list:
    """Sắp xếp tự nhiên: CV_2 < CV_10 < CV_100 (thay vì CV_10 < CV_100 < CV_2)."""
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def cv_group(stem: str) -> str:
    """'CV_102_page_002' → 'cv_102'. Dùng để gom các trang của cùng một CV."""
    import re

    m = re.match(r"(?i)^(cv[_ -]?\d+)", stem)
    if m:
        return m.group(1).lower().replace(" ", "_").replace("-", "_")
    m = re.match(r"(?i)^(.*?)[_ -](?:page|p)[_ -]?\d+$", stem)
    return (m.group(1) if m else stem).lower()


def page_number(stem: str) -> int:
    """'CV_102_page_002' → 2 (không có số trang → 1)."""
    import re

    m = re.search(r"(?i)(?:page|p)[_ -]?(\d+)$", stem)
    return int(m.group(1)) if m else 1


def active_weights() -> Path:
    """Trọng số ML backend / demo dùng: env YOLO_WEIGHTS > weights/finetune.pt > BEST_WEIGHTS.
    Gọi lại mỗi request -> khi retrain tạo finetune.pt, backend tự chuyển sang."""
    env = os.environ.get("YOLO_WEIGHTS")
    if env:
        return Path(env)
    return FINETUNE_WEIGHTS if FINETUNE_WEIGHTS.exists() else BEST_WEIGHTS


def auto_device() -> str:
    """'0' nếu có CUDA, ngược lại 'cpu' (tránh crash khi máy không có GPU)."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
