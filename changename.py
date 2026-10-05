"""
Đổi tên PDF thành CV_1.pdf, CV_2.pdf, ...

An toàn khi chạy lại (idempotent):
  - File đã đúng dạng CV_<số>.pdf được GIỮ NGUYÊN.
  - Chỉ file mới được đánh số tiếp theo (max + 1, max + 2, ...).
  - Ghi log ánh xạ tên cũ → tên mới vào rename_log.csv.

Bản cũ sắp xếp theo chuỗi ("CV_1, CV_10, CV_100, CV_2...") và đánh số lại
toàn bộ → chạy lần 2 sẽ tráo tên CV, làm lệch toàn bộ nhãn Label Studio.

Usage:
    python changename.py              # đổi tên file mới
    python changename.py --dry-run    # chỉ xem trước
    python changename.py --renumber   # đánh số lại TOÀN BỘ (natural sort) – cẩn thận!
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from config import PDF_DIR, natural_key

CV_PATTERN = re.compile(r"^CV_(\d+)\.pdf$", re.IGNORECASE)


def plan_renames(pdf_dir: Path, renumber: bool):
    pdfs = sorted((p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
                  key=natural_key)

    if renumber:
        return [(p, pdf_dir / f"CV_{i}.pdf") for i, p in enumerate(pdfs, 1)]

    existing = [p for p in pdfs if CV_PATTERN.match(p.name)]
    new = [p for p in pdfs if not CV_PATTERN.match(p.name)]
    start = max((int(CV_PATTERN.match(p.name).group(1)) for p in existing), default=0)
    return [(p, pdf_dir / f"CV_{start + i}.pdf") for i, p in enumerate(new, 1)]


def apply_renames(plan):
    # 2 pha qua tên tạm → không bao giờ ghi đè khi tên cũ/mới giao nhau
    temps = []
    for i, (src, dst) in enumerate(plan):
        tmp = src.with_name(f"__tmp_rename_{i}__.pdf")
        src.rename(tmp)
        temps.append((tmp, dst))
    for tmp, dst in temps:
        tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=PDF_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--renumber", action="store_true",
                    help="Đánh số lại tất cả. KHÔNG dùng sau khi đã gán nhãn.")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"❌ Không tìm thấy thư mục: {args.dir}")

    plan = [(s, d) for s, d in plan_renames(args.dir, args.renumber) if s.name != d.name]
    if not plan:
        print("✓ Không có file nào cần đổi tên.")
        return

    for src, dst in plan:
        print(f"{src.name:<50} → {dst.name}")
    print(f"\nTổng: {len(plan)} file")

    if args.dry_run:
        print("(dry-run: chưa đổi gì)")
        return

    apply_renames(plan)

    log = args.dir / "rename_log.csv"
    is_new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["time", "old_name", "new_name"])
        now = datetime.now().isoformat(timespec="seconds")
        w.writerows((now, s.name, d.name) for s, d in plan)

    print(f"✓ Đã đổi tên {len(plan)} file. Log: {log}")


if __name__ == "__main__":
    main()
