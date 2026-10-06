"""Chia train/val/test THEO CV (group) và sinh data.yaml cho Ultralytics.

python training/split_dataset.py                       (src/dst mặc định từ config.py)
python training/split_dataset.py --overwrite           (xoá DATASET_DIR cũ rồi chia lại)

Ngoài data.yaml còn ghi:
  stats.csv    số box mỗi lớp theo từng tập (đưa vào báo cáo)
  split.json   tham số (seed, tỉ lệ, src) + danh sách CV của từng tập (tái lập được)
  vis/         --vis ảnh ngẫu nhiên có vẽ box, để kiểm tra tọa độ bằng mắt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, LS_ROUNDS_DIR, natural_key  # noqa: E402

# Distinct colors for box drawing (cycled if there are more classes)
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#469990", "#800000", "#808000", "#000075", "#a9a9a9"]


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def draw_boxes(img_path: Path, lbl_path: Path, names: list[str], dst: Path) -> None:
    """Draw YOLO boxes back onto the image (pixel coords recomputed from normalized values)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 500)
    try:
        font = ImageFont.load_default(size=max(14, W // 60))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for c, cx, cy, w, h in read_labels(lbl_path):
        x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        color = PALETTE[c % len(PALETTE)]
        name = names[c] if 0 <= c < len(names) else f"CLASS_{c}"
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        tb = d.textbbox((x1, y1), name, font=font)
        d.rectangle([tb[0], tb[1] - (tb[3] - tb[1]) - 4, tb[2] + 4, tb[1]], fill=color)
        d.text((x1 + 2, tb[1] - (tb[3] - tb[1]) - 3), name, fill="white", font=font)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(LS_ROUNDS_DIR / "raw"))
    ap.add_argument("--dst", default=str(DATASET_DIR))
    ap.add_argument("--overwrite", action="store_true", help="Xoá dst cũ trước khi chia")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vis", type=int, default=5, help="Số ảnh ngẫu nhiên vẽ box để kiểm tra (0 = tắt)")
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

    keys = sorted(groups, key=natural_key)  # deterministic order before the seeded shuffle
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    n_tr = round(n * args.ratios[0])
    n_va = max(1, round(n * args.ratios[1])) if n >= 3 else 0
    split_keys = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    if not split_keys["val"]:
        split_keys["val"] = split_keys["train"][-1:]

    # Leakage check: no CV (and therefore no page) may appear in two splits
    owner = {}
    for split, ks in split_keys.items():
        for k in ks:
            if k in owner and {owner[k], split} != {"train", "val"}:  # train/val overlap only when n < 3
                raise SystemExit(f"❌ Rò rỉ dữ liệu: CV {k} nằm ở cả {owner[k]} và {split}")
            owner.setdefault(k, split)

    names = [c.strip() for c in (src / "classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    stats = {s: Counter() for s in split_keys}
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
                stats[split].update(c for c, *_ in read_labels(lbl))
        print(f"{split:5s}: {len(ks):4d} CV, {sum(len(groups[k]) for k in ks):5d} trang")

    if not split_keys["test"]:
        print("  ! Tập test rỗng (quá ít CV) – evaluate.py sẽ không chạy được với --split test")

    # Per-class box counts per split
    splits = list(split_keys)
    header = ["class_id", "class", *splits, "total"]
    rows = [[i, nm, *(stats[s][i] for s in splits), sum(stats[s][i] for s in splits)]
            for i, nm in enumerate(names)]
    rows.append(["", "TOTAL", *(sum(stats[s].values()) for s in splits),
                 sum(sum(stats[s].values()) for s in splits)])
    with open(dst / "stats.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    print("\nSố box mỗi lớp:")
    print(f"{'id':>3} {'Lớp':<15}" + "".join(f"{s:>8}" for s in [*splits, "total"]))
    for r in rows:
        print(f"{r[0]!s:>3} {r[1]:<15}" + "".join(f"{v:>8}" for v in r[2:]))
    for i, nm in enumerate(names):
        empty = [s for s in splits if stats[s][i] == 0]
        if empty:
            print(f"  ! Lớp {nm} không có box trong: {', '.join(empty)}")

    data = {"path": str(dst.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: n for i, n in enumerate(names)}}
    (dst / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (dst / "classes.txt").write_text("\n".join(names), encoding="utf-8")
    log = {"src": str(src.resolve()), "seed": args.seed, "ratios": list(args.ratios),
           "n_cv": {s: len(ks) for s, ks in split_keys.items()},
           "n_pages": {s: sum(len(groups[k]) for k in ks) for s, ks in split_keys.items()},
           "cv": {s: sorted(ks, key=natural_key) for s, ks in split_keys.items()}}
    (dst / "split.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {dst / 'data.yaml'}, stats.csv, split.json")

    if args.vis > 0:
        (dst / "vis").mkdir(exist_ok=True)
        pool = sorted(((s, p) for s in splits for p in (dst / "images" / s).iterdir()),
                      key=lambda t: natural_key(t[1].name))
        for s, p in random.Random(args.seed).sample(pool, min(args.vis, len(pool))):
            draw_boxes(p, dst / "labels" / s / f"{p.stem}.txt", names, dst / "vis" / f"{s}_{p.stem}.jpg")
        print(f"-> {dst / 'vis'} ({min(args.vis, len(pool))} ảnh kiểm tra)")


if __name__ == "__main__":
    main()
