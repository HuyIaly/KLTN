"""So sánh các model trên tập test: mAP tổng + AP từng lớp + tốc độ.

python training/evaluate.py          (mặc định: so sánh WEIGHTS_DIR/scratch.pt và finetune.pt)
python training/evaluate.py --models scratch=... finetune=... old=D:\\Final\\runs\\cv_layout_yolo\\weights\\best.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATASET_DIR, FINETUNE_WEIGHTS, IMGSZ, RUNS_DIR,  # noqa: E402
                    SCRATCH_WEIGHTS, auto_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--models", nargs="+", help="ten=duong_dan.pt",
                    default=[f"scratch={SCRATCH_WEIGHTS}", f"finetune={FINETUNE_WEIGHTS}"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(RUNS_DIR / "eval"))
    args = ap.parse_args()

    from ultralytics import YOLO

    args.device = args.device or auto_device()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overall, per_class = [], []
    for spec in args.models:
        name, path = spec.split("=", 1)
        m = YOLO(path)
        met = m.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, conf=0.001, iou=0.6, plots=True,
                    project=str(out), name=name, exist_ok=True, verbose=False)
        b = met.box
        overall.append({"model": name, "precision": b.mp, "recall": b.mr, "mAP50": b.map50,
                        "mAP50-95": b.map, "ms/img": met.speed.get("inference", float("nan"))})
        for j, c in enumerate(b.ap_class_index):
            p, r, ap50, ap5095 = b.class_result(j)
            per_class.append({"model": name, "class": met.names[int(c)], "precision": p,
                              "recall": r, "AP50": ap50, "AP50-95": ap5095})

    for fname, rows in (("overall.csv", overall), ("per_class.csv", per_class)):
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\n| Model | P | R | mAP50 | mAP50-95 | ms/img |\n|---|---|---|---|---|---|")
    for r in overall:
        print(f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} "
              f"| {r['mAP50-95']:.3f} | {r['ms/img']:.1f} |")

    names = [r["model"] for r in overall]
    classes = sorted({r["class"] for r in per_class})
    print("\nAP50-95 theo lớp:\n| Lớp | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for c in classes:
        vals = []
        for n in names:
            v = next((r["AP50-95"] for r in per_class if r["model"] == n and r["class"] == c), None)
            vals.append(f"{v:.3f}" if v is not None else "-")
        print(f"| {c} | " + " | ".join(vals) + " |")
    print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()
