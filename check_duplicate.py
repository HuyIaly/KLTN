"""
Tìm PDF trùng lặp.

  1. Trùng tuyệt đối (byte-level): chỉ hash các file CÓ CÙNG KÍCH THƯỚC
     → file có size duy nhất không cần đọc.
  2. Trùng nội dung (text giống nhau, khác metadata/spacing).

Chạy TRƯỚC khi chia train/val/test: CV trùng nằm ở 2 split sẽ làm
mAP trên val/test bị ảo.

Usage:
    python check_duplicate.py [--dir D:\\Final\\CV] [--no-content]
"""

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from config import PDF_DIR, natural_key

try:
    import pymupdf as fitz
except ImportError:
    import fitz

_WS = re.compile(r"\s+")


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def content_hash(path: Path):
    try:
        with fitz.open(path) as doc:
            text = " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"\n[ERROR] {path.name}: {e}")
        return None
    text = _WS.sub(" ", text).strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def groups_of(mapping):
    return [sorted(files, key=natural_key) for files in mapping.values() if len(files) > 1]


def exact_duplicates(pdfs):
    by_size = defaultdict(list)
    for p in pdfs:
        by_size[p.stat().st_size].append(p)

    by_hash = defaultdict(list)
    for same_size in by_size.values():
        if len(same_size) > 1:
            for p in same_size:
                by_hash[sha256_file(p)].append(p)
    return groups_of(by_hash)


def content_duplicates(pdfs, skip):
    by_text = defaultdict(list)
    no_text = []
    for i, p in enumerate(pdfs, 1):
        print(f"\r  Extract {i}/{len(pdfs)}", end="", flush=True)
        if p in skip:
            continue
        h = content_hash(p)
        (by_text[h].append(p) if h else no_text.append(p))
    print()
    return groups_of(by_text), no_text


def report(title, groups):
    print(f"\n{title}: {len(groups)} nhóm")
    for i, files in enumerate(groups, 1):
        print(f"  Nhóm {i}: giữ {files[0].name}  |  trùng: {', '.join(f.name for f in files[1:])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=PDF_DIR)
    ap.add_argument("--no-content", action="store_true", help="Bỏ bước so sánh text")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Folder does not exist: {args.dir}")

    pdfs = sorted((p for p in args.dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
                  key=natural_key)
    print("=" * 70)
    print(f"PDF DUPLICATE CHECKER  |  {args.dir}  |  {len(pdfs)} PDF")
    print("=" * 70)

    exact = exact_duplicates(pdfs)
    report("[1/2] Trùng tuyệt đối", exact)

    if not args.no_content:
        # Bản sao byte-level chỉ cần trích text 1 lần (bỏ các bản sao phụ)
        skip = {p for g in exact for p in g[1:]}
        content, no_text = content_duplicates(pdfs, skip)
        report("[2/2] Trùng nội dung", content)
        if no_text:
            print(f"\n⚠ {len(no_text)} PDF không có text layer (scan?) → không so sánh được nội dung:")
            print("  " + ", ".join(p.name for p in no_text[:20]))


if __name__ == "__main__":
    main()
