"""Chuyển JSON export của Label Studio -> định dạng YOLO detection.

Hỗ trợ:
  * Template Multi-page Document Annotation (1 task = 1 CV, data là list ảnh trang,
    mỗi box có `item_index` = số thứ tự trang).
  * Template ảnh đơn (data là 1 URL ảnh).

Mặc định đọc đường dẫn + CLASS_NAMES từ config.py:
  python training/labelstudio_to_yolo.py --export D:\\Final\\labelstudio\\export.json

Đầu ra (--out, mặc định BASE_DIR/ls_rounds/raw):
  images/*.png (GIỮ tên gốc, vd CV_12_page_001.png), labels/*.txt, classes.txt,
  groups.csv (ảnh -> CV theo config.cv_group) để chia train/val/test THEO CV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, LS_ROUNDS_DIR, PNG_DIR, cv_group  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def url_to_name(url: str) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    if "d" in q:  # /data/local-files/?d=Final/pages/cv01_p1.png
        return Path(unquote(q["d"][0])).name
    return Path(unquote(u.path)).name


def build_index(images_dir: Path) -> dict:
    return {p.name: p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXT}


def resolve(url: str, index: dict) -> Path | None:
    p = Path(unquote(url))
    if p.exists():
        return p
    name = url_to_name(url)
    if name in index:
        return index[name]
    m = re.match(r"^[0-9a-f]{6,}-(.+)$", name)  # file upload: "<hash>-ten_goc.png"
    if m and m.group(1) in index:
        return index[m.group(1)]
    return None


def page_urls(data: dict, key: str | None) -> list[str]:
    if key:
        v = data[key]
        return v if isinstance(v, list) else [v]
    for v in data.values():  # tự đoán: list -> multi-page
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v
    for v in data.values():
        if isinstance(v, str) and (Path(url_to_name(v)).suffix.lower() in IMG_EXT or "/data/" in v):
            return [v]
    raise KeyError(f"Không tìm thấy trường ảnh trong data: {list(data)}")


def pick_annotation(task: dict) -> dict | None:
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    return max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="File JSON export (định dạng JSON, không phải JSON-MIN)")
    ap.add_argument("--images-dir", default=str(PNG_DIR), help="Thư mục ảnh trang (mặc định PNG_DIR)")
    ap.add_argument("--out", default=str(LS_ROUNDS_DIR / "raw"))
    ap.add_argument("--classes", default=None, help="classes.txt; bỏ trống = CLASS_NAMES trong config.py")
    ap.add_argument("--data-key", default=None, help="Tên trường ảnh trong task.data (vd pages, image)")
    ap.add_argument("--include-empty", action="store_true", help="Giữ trang không có box làm ảnh âm")
    args = ap.parse_args()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    index = build_index(Path(args.images_dir))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    if args.classes:
        classes = [c.strip() for c in Path(args.classes).read_text(encoding="utf-8").splitlines() if c.strip()]
    else:
        classes = list(CLASS_NAMES)
    cid = {c: i for i, c in enumerate(classes)}

    groups, n_img, n_box, missing, unknown, rotated = [], 0, 0, 0, set(), 0
    for t in tasks:
        ann = pick_annotation(t)
        if ann is None:
            continue
        urls = page_urls(t["data"], args.data_key)
        per_page: dict[int, list[str]] = {i: [] for i in range(len(urls))}
        for r in ann.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            label = (v.get("rectanglelabels") or [None])[0]
            if label not in cid:
                unknown.add(label)
                continue
            if abs(v.get("rotation", 0) or 0) > 0.5:
                rotated += 1
            k = int(r.get("item_index", 0) or 0)
            x, y, w, h = (float(v[c]) / 100.0 for c in ("x", "y", "width", "height"))
            cx, cy = min(max(x + w / 2, 0), 1), min(max(y + h / 2, 0), 1)
            w, h = min(max(w, 0), 1), min(max(h, 0), 1)
            if w <= 0 or h <= 0:
                continue
            per_page.setdefault(k, []).append(f"{cid[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        for k, url in enumerate(urls):
            lines = per_page.get(k, [])
            if not lines and not args.include_empty:
                continue
            src = resolve(url, index)
            if src is None:
                missing += 1
                print(f"[THIẾU ẢNH] task {t.get('id')} trang {k}: {url}")
                continue
            stem = src.stem  # giữ tên gốc -> truy ngược được về PDF
            shutil.copy2(src, out / "images" / f"{stem}{src.suffix.lower()}")
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            groups.append((f"{stem}{src.suffix.lower()}", cv_group(stem)))
            n_img += 1
            n_box += len(lines)

    (out / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    with open(out / "groups.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("image", "group")] + groups)

    print(f"Xong: {n_img} ảnh, {n_box} box, {len(classes)} lớp -> {out}")
    if missing:
        print(f"  ! {missing} trang không tìm thấy ảnh (kiểm tra --images-dir)")
    if unknown:
        print(f"  ! Bỏ qua nhãn không có trong classes: {sorted(unknown)}")
    if rotated:
        print(f"  ! {rotated} box có rotation != 0 (đã bỏ qua góc xoay)")


if __name__ == "__main__":
    main()
