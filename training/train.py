"""Huấn luyện YOLO layout cho CV – 2 chế độ để so sánh:

  scratch  : khởi tạo ngẫu nhiên từ file kiến trúc .yaml (KHÔNG dùng trọng số pretrained)
  finetune : khởi tạo từ trọng số pretrained (.pt COCO mặc định, hoặc trọng số của bạn,
             vd model đã train trên DocLayNet) rồi tinh chỉnh trên dữ liệu CV

Ví dụ:
  python training/train.py --mode scratch  --arch yolo11s
  python training/train.py --mode finetune --arch yolo11s
  python training/train.py --mode finetune --weights path/doclaynet_best.pt --freeze 10

Đường dẫn mặc định lấy từ config.py; best.pt được copy sang WEIGHTS_DIR/<mode>.pt
(D:\\Final\\weights\\scratch.pt | finetune.pt) để demo, CLI và ML backend dùng.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, IMGSZ, RUNS_DIR, WEIGHTS_DIR, auto_device  # noqa: E402

# Augmentation hợp với tài liệu: KHÔNG lật (chữ bị ngược), không xoay/shear,
# mosaic nhẹ (mosaic mạnh phá ngữ cảnh bố cục trang), đổi màu nhẹ (CV có nhiều theme màu).
DOC_AUG = dict(
    fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    translate=0.05, scale=0.25, mosaic=0.3, close_mosaic=15, mixup=0.0, copy_paste=0.0,
    hsv_h=0.015, hsv_s=0.3, hsv_v=0.3,
)

MODE_DEFAULTS = {
    # Từ đầu: cần nhiều epoch hơn, LR cao hơn, warmup dài, patience lớn.
    "scratch": dict(epochs=300, optimizer="SGD", lr0=0.01, momentum=0.937, warmup_epochs=5, patience=100),
    # Finetune: ít epoch hơn, LR thấp hơn để không phá đặc trưng pretrained.
    "finetune": dict(epochs=100, optimizer="AdamW", lr0=0.001, warmup_epochs=3, patience=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "finetune"], required=True)
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--arch", default="yolo11s", help="yolo11n/s/m/l/x hoặc yolov8s...")
    ap.add_argument("--weights", default=None, help="finetune: trọng số khởi tạo (mặc định <arch>.pt)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="0 | 0,1 | cpu (mặc định tự chọn)")
    ap.add_argument("--workers", type=int, default=2, help="Windows nên để thấp (0-4)")
    ap.add_argument("--freeze", type=int, default=0, help="finetune: đóng băng N layer đầu (backbone)")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--project", default=str(RUNS_DIR))
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-as", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from ultralytics import YOLO

    args.device = args.device or auto_device()

    hp = dict(MODE_DEFAULTS[args.mode])
    if args.epochs:
        hp["epochs"] = args.epochs
    if args.lr0:
        hp["lr0"] = args.lr0

    if args.mode == "scratch":
        model = YOLO(f"{args.arch}.yaml")          # kiến trúc rỗng, trọng số ngẫu nhiên
        hp["pretrained"] = False
    else:
        model = YOLO(args.weights or f"{args.arch}.pt")  # Ultralytics tự thay head theo số lớp mới
        hp["pretrained"] = True
        if args.freeze > 0:
            hp["freeze"] = args.freeze

    name = args.name or f"{args.mode}_{args.arch}_{args.imgsz}"
    model.train(
        data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
        project=args.project, name=name, seed=args.seed, deterministic=True, cos_lr=True,
        plots=True, exist_ok=False, **DOC_AUG, **hp,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    dst = Path(args.save_as or WEIGHTS_DIR / f"{args.mode}.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    (dst.with_suffix(".json")).write_text(json.dumps({
        "mode": args.mode, "arch": args.arch, "init": args.weights or (f"{args.arch}.pt" if args.mode == "finetune" else None),
        "imgsz": args.imgsz, "run_dir": str(save_dir), **{k: v for k, v in hp.items() if k != "pretrained"},
    }, indent=2), encoding="utf-8")
    print(f"\nBest: {best}\nĐã copy -> {dst}")


if __name__ == "__main__":  # bắt buộc trên Windows (dataloader dùng multiprocessing)
    main()
