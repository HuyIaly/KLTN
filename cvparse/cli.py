"""Chạy pipeline từ dòng lệnh.

python -m cvparse.cli path/to/cv.pdf --weights weights/finetune.pt --out outputs
python -m cvparse.cli data/test_cvs/ --weights weights/scratch.pt --text-mode ocr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

from .normalize import IMAGE_EXTS, OFFICE_EXTS
from .pipeline import CVLayoutPipeline, PipelineConfig

SUPPORTED = IMAGE_EXTS | OFFICE_EXTS | {".pdf"}


def _defaults():
    """Lấy mặc định từ config.py ở gốc project nếu có (package vẫn chạy độc lập khi không có)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        import config
        return dict(weights=str(config.active_weights()), imgsz=config.IMGSZ, dpi=config.RENDER_DPI,
                    out=str(config.BASE_DIR / "outputs"), device=config.auto_device())
    except ImportError:  # running without the project's config.py
        return dict(weights=None, imgsz=1024, dpi=200, out="outputs", device=None)


def main():
    d = _defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--weights", default=d["weights"], required=d["weights"] is None)
    ap.add_argument("--out", default=d["out"])
    ap.add_argument("--device", default=d["device"], help="0 | cpu | ...")
    ap.add_argument("--imgsz", type=int, default=d["imgsz"])
    ap.add_argument("--dpi", type=int, default=d["dpi"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--text-mode", choices=["auto", "text_layer", "ocr"], default="auto")
    ap.add_argument("--ocr", choices=["easyocr", "tesseract"], default="easyocr")
    ap.add_argument("--order", choices=["x_first", "y_first"], default="x_first")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.inputs):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]

    pipe = CVLayoutPipeline(args.weights, device=args.device, imgsz=args.imgsz, ocr_engine=args.ocr)
    cfg = PipelineConfig(dpi=args.dpi, conf=args.conf, iou=args.iou,
                         text_mode=args.text_mode, order_strategy=args.order)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            res = pipe.run(f, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[LỖI] {f}: {e}")
            continue
        (out / f"{f.stem}.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        if not args.no_viz:
            for i, im in enumerate(res.annotated_images()):
                im.save(out / f"{f.stem}_p{i + 1}.png")
        print(f"[OK] {f.name}: {len(res.sections)} mục, {res.timings['total']:.2f}s")


if __name__ == "__main__":
    main()
