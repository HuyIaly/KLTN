"""Chia train/val/test THEO CV (group) và sinh data.yaml cho Ultralytics.

python training/split_dataset.py                       (src/dst mặc định từ config.py)
python training/split_dataset.py --overwrite           (xoá DATASET_DIR cũ rồi chia lại)
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, LS_ROUNDS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(LS_ROUNDS_DIR / "raw"))
    ap.add_argument("--dst", default=str(DATASET_DIR))
    ap.add_argument("--overwrite", action="store_true", help="Xoá dst cũ trước khi chia")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        if not args.overwrite:
            # Copy chồng lên dataset cũ -> ảnh cũ còn sót ở split khác -> rò rỉ train/test
            raise SystemExit(f"❌ {dst} đã có dữ liệu. Thêm --overwrite để xoá và chia lại.")
        shutil.rmtree(dst)
    groups = defaultdict(list)
    gfile = src / "groups.csv"
    if gfile.exists():
        with open(gfile, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                groups[row["group"]].append(row["image"])
    else:  # không có groups.csv -> mỗi ảnh là 1 nhóm
        for p in (src / "images").iterdir():
            groups[p.stem].append(p.name)

    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    n_tr = round(n * args.ratios[0])
    n_va = max(1, round(n * args.ratios[1])) if n >= 3 else 0
    split_keys = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    if not split_keys["val"]:
        split_keys["val"] = split_keys["train"][-1:]

    for split, ks in split_keys.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k in ks:
            for img in groups[k]:
                stem = Path(img).stem
                shutil.copy2(src / "images" / img, dst / "images" / split / img)
                lbl = src / "labels" / f"{stem}.txt"
                (dst / "labels" / split / f"{stem}.txt").write_text(
                    lbl.read_text(encoding="utf-8") if lbl.exists() else "", encoding="utf-8")
        print(f"{split:5s}: {len(ks):4d} CV, {sum(len(groups[k]) for k in ks):5d} trang")

    if not split_keys["test"]:
        print("  ! Tập test rỗng (quá ít CV) – evaluate.py sẽ không chạy được với --split test")
    names = [c.strip() for c in (src / "classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    data = {"path": str(dst.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: n for i, n in enumerate(names)}}
    (dst / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"-> {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
