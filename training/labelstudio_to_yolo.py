"""Chuyển JSON export của Label Studio -> định dạng YOLO detection.

Hỗ trợ:
  * Template Multi-page Document Annotation (1 task = 1 CV, data là list ảnh trang,
    mỗi box có `item_index` = số thứ tự trang).
  * Template ảnh đơn (data là 1 URL ảnh, 1 task = 1 trang).

Mặc định đọc đường dẫn + CLASS_NAMES từ config.py:
  python training/labelstudio_to_yolo.py --export D:\\Final\\project.json

Tách trang: nếu không tìm thấy ảnh trang trong --images-dir, script render trang đó
từ PDF gốc (--pdf-dir, tên dạng CV_12_page_003.png -> CV_12.pdf trang 3) ở RENDER_DPI.

Đầu ra (--out, mặc định BASE_DIR/ls_rounds/raw):
  images/*.png (GIỮ tên gốc, vd CV_12_page_001.png), labels/*.txt, classes.txt,
  groups.csv (ảnh -> CV theo config.cv_group) để chia train/val/test THEO CV
  (bước chia: training/split_dataset.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, LS_ROUNDS_DIR, PDF_DIR, PNG_DIR, RENDER_DPI, cv_group  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def url_to_name(url: str) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    if "d" in q:  # /data/local-files/?d=PNG%5CCV_1%5CCV_1_page_001.png
        return Path(unquote(q["d"][0]).replace("\\", "/")).name
    return Path(unquote(u.path).replace("\\", "/")).name


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


def render_from_pdf(name: str, pdf_dir: Path, dst: Path, dpi: int) -> bool:
    """Render 'CV_12_page_003.png' from CV_12.pdf (page 3) into dst. Return False if impossible."""
    m = re.match(r"(?i)^(.+?)_page_(\d+)$", Path(name).stem)
    if not m:
        return False
    pdf = pdf_dir / f"{m.group(1)}.pdf"
    if not pdf.exists():
        return False
    import fitz  # pymupdf, imported lazily: only needed when a page image is missing

    with fitz.open(pdf) as doc:
        k = int(m.group(2)) - 1
        if not 0 <= k < doc.page_count:
            return False
        doc[k].get_pixmap(dpi=dpi).save(dst)
    return True


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


def to_yolo(v: dict) -> tuple[float, float, float, float] | None:
    """Label Studio percent (top-left x, y, w, h) -> YOLO normalized (cx, cy, w, h).
    Clip the box corners to the image, so a box spilling over an edge keeps its visible part."""
    x1, y1 = float(v["x"]) / 100.0, float(v["y"]) / 100.0
    x2, y2 = x1 + float(v["width"]) / 100.0, y1 + float(v["height"]) / 100.0
    x1, y1, x2, y2 = (min(max(c, 0.0), 1.0) for c in (x1, y1, x2, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 1e-4 or h <= 1e-4:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="File JSON export (định dạng JSON, không phải JSON-MIN)")
    ap.add_argument("--images-dir", default=str(PNG_DIR), help="Thư mục ảnh trang (mặc định PNG_DIR)")
    ap.add_argument("--pdf-dir", default=str(PDF_DIR), help="PDF gốc để render trang còn thiếu ảnh")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--out", default=str(LS_ROUNDS_DIR / "raw"))
    ap.add_argument("--overwrite", action="store_true", help="Xoá --out cũ trước khi ghi")
    ap.add_argument("--classes", default=None, help="classes.txt; bỏ trống = CLASS_NAMES trong config.py")
    ap.add_argument("--data-key", default=None, help="Tên trường ảnh trong task.data (vd pages, image)")
    ap.add_argument("--include-empty", action="store_true", help="Giữ trang không có box làm ảnh âm")
    args = ap.parse_args()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    index = build_index(Path(args.images_dir))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            # Stale images from a previous export would silently leak into the new dataset
            raise SystemExit(f"❌ {out} đã có dữ liệu. Thêm --overwrite để xoá và tạo lại.")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    if args.classes:
        classes = [c.strip() for c in Path(args.classes).read_text(encoding="utf-8").splitlines() if c.strip()]
    else:
        classes = list(CLASS_NAMES)
    cid = {c: i for i, c in enumerate(classes)}

    groups, missing, rotated, dropped, rendered = [], 0, 0, 0, 0
    unknown, per_class, seen = Counter(), Counter(), {}
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
                unknown[label] += 1
                continue
            if abs(v.get("rotation", 0) or 0) > 0.5:
                rotated += 1
            box = to_yolo(v)
            if box is None:
                dropped += 1
                continue
            k = int(r.get("item_index", 0) or 0)
            per_page.setdefault(k, []).append(f"{cid[label]} " + " ".join(f"{c:.6f}" for c in box))
            per_class[label] += 1

        for k, url in enumerate(urls):
            lines = per_page.get(k, [])
            if not lines and not args.include_empty:
                continue
            src = resolve(url, index)
            name = src.name if src else url_to_name(url)
            stem, suffix = Path(name).stem, Path(name).suffix.lower() or ".png"
            dst = out / "images" / f"{stem}{suffix}"
            if stem in seen:
                print(f"[TRÙNG TRANG] {stem}: task {seen[stem]} và task {t.get('id')} -> giữ task sau")
            if src is not None:
                shutil.copy2(src, dst)
            elif render_from_pdf(name, Path(args.pdf_dir), dst, args.dpi):
                rendered += 1
            else:
                missing += 1
                print(f"[THIẾU ẢNH] task {t.get('id')} trang {k}: {url}")
                continue
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if stem not in seen:
                groups.append((dst.name, cv_group(stem)))
            seen[stem] = t.get("id")

    (out / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    with open(out / "groups.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("image", "group")] + groups)

    n_box = sum(per_class.values())
    n_cv = len({g for _, g in groups})
    print(f"Xong: {len(groups)} ảnh, {n_cv} CV, {n_box} box, {len(classes)} lớp -> {out}")
    print(f"{'Lớp':<15}{'id':>4}{'Số box':>9}")
    for c in classes:
        print(f"{c:<15}{cid[c]:>4}{per_class[c]:>9}")
    if rendered:
        print(f"  i {rendered} trang được render từ PDF ({args.dpi} DPI)")
    if missing:
        print(f"  ! {missing} trang không tìm thấy ảnh/PDF (kiểm tra --images-dir, --pdf-dir)")
    if unknown:
        print(f"  ! Bỏ qua nhãn không có trong classes: {dict(unknown)}")
    if dropped:
        print(f"  ! Bỏ {dropped} box rỗng hoặc nằm ngoài ảnh")
    if rotated:
        print(f"  ! {rotated} box có rotation != 0 (đã bỏ qua góc xoay)")
    if not any(per_class[c] for c in classes):
        raise SystemExit("❌ Không có box nào khớp classes – kiểm tra tên nhãn.")


if __name__ == "__main__":
    main()
