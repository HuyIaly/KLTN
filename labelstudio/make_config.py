"""Sinh label config XML từ CLASS_NAMES trong config.py (thứ tự lớp khớp id YOLO).

python labelstudio/make_config.py                  -> D:\\Final\\labelstudio\\label_config.xml
python labelstudio/make_config.py --single-image   (1 task = 1 trang, data key "image")
Dán nội dung file vào Project > Settings > Labeling Interface > Code.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, LS_DIR  # noqa: E402
from labelstudio.core import build_label_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(LS_DIR / "label_config.xml"))
ap.add_argument("--single-image", action="store_true")
a = ap.parse_args()

xml = build_label_config(CLASS_NAMES, multipage=not a.single_image,
                         value_key="image" if a.single_image else "pages")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(xml, encoding="utf-8")
print(xml)
print(f"-> {a.out}")
