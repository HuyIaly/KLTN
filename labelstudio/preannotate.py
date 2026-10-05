"""Pre-annotate theo lô (offline, không cần chạy ML backend).

PDF trong PDF_DIR -> ảnh trang trong PNG_DIR (chỉ render trang CHƯA có, tên CV_1_page_001.png,
RENDER_DPI) -> YOLO -> file tasks JSON có sẵn `predictions` -> Import vào Label Studio.

python labelstudio/preannotate.py                                   # mọi CV
python labelstudio/preannotate.py --skip-labeled D:\\Final\\labelstudio\\export.json
                                   # bỏ các CV đã có trong project (chỉ CV mới sau changename.py)
python labelstudio/preannotate.py --inputs D:\\Final\\CV\\CV_120.pdf D:\\Final\\CV\\CV_121.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (BASE_DIR, IMAGE_EXTS, IMGSZ, LS_DIR, PAGE_NAME, PDF_DIR,  # noqa: E402
                    PNG_DIR, RENDER_DPI, active_weights, auto_device, cv_group,
                    natural_key, page_number)
from labelstudio.core import (CACHE, detect, local_files_url, open_rgb,  # noqa: E402
                              parse_label_config, task_score, to_ls_results, url_to_name)

DOC_EXT = {".pdf", ".docx", ".doc"}


def labeled_groups(export_path: Path) -> set:
    """Các CV đã nằm trong project (đọc từ file export JSON)."""
    groups = set()
    for t in json.loads(export_path.read_text(encoding="utf-8")):
        for v in t.get("data", {}).values():
            for url in (v if isinstance(v, list) else [v]):
                if isinstance(url, str) and url.strip():
                    groups.add(cv_group(Path(url_to_name(url)).stem))
    return groups


def collect_pages(inputs, pages_dir: Path, dpi: int, skip: set) -> dict:
    """{cv_group: [ảnh trang theo thứ tự]}. PDF chỉ render những trang chưa có PNG."""
    from PIL import Image

    groups = defaultdict(dict)
    files = []
    for p in map(Path, inputs):
        files += sorted((f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS | DOC_EXT), key=natural_key) \
            if p.is_dir() else [p]

    for f in files:
        g = cv_group(f.stem)
        if g in skip:
            continue
        if f.suffix.lower() in DOC_EXT:
            existing = sorted(pages_dir.glob(f"{f.stem}_page_*.png"), key=natural_key)
            if existing:  # đã render trước đó -> dùng lại, đảm bảo khớp ảnh đã gán nhãn
                for img in existing:
                    groups[g].setdefault(page_number(img.stem), img)
                continue
            from cvparse.normalize import load_document
            doc = load_document(f, dpi=dpi, max_pages=20)
            try:
                pages_dir.mkdir(parents=True, exist_ok=True)
                for pg in doc.pages:
                    out = pages_dir / PAGE_NAME.format(stem=f.stem, page=pg.index + 1)
                    Image.fromarray(pg.image).save(out)
                    groups[g].setdefault(pg.index + 1, out)
            finally:
                doc.close()
        else:
            groups[g].setdefault(page_number(f.stem), f)
    return {g: [v[k] for k in sorted(v)] for g, v in sorted(groups.items(), key=lambda kv: natural_key(kv[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[str(PDF_DIR)], help="PDF/ảnh hoặc thư mục (mặc định PDF_DIR)")
    ap.add_argument("--pages-dir", default=str(PNG_DIR))
    ap.add_argument("--doc-root", default=str(BASE_DIR), help="= LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
    ap.add_argument("--weights", default=None, help="mặc định config.active_weights()")
    ap.add_argument("--label-config", default=str(LS_DIR / "label_config.xml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-labeled", default=None, help="File export JSON: bỏ các CV đã có trong project")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label-map", default="{}", help='JSON, vd {"EXP":"EXPERIENCE"}')
    args = ap.parse_args()

    weights = str(args.weights or active_weights())
    if not Path(weights).exists():
        raise SystemExit(f"❌ Không tìm thấy trọng số: {weights}")
    if not Path(args.label_config).exists():
        raise SystemExit(f"❌ Chưa có label config: {args.label_config}\n   Chạy: python labelstudio/make_config.py")

    cfg = parse_label_config(Path(args.label_config).read_text(encoding="utf-8"))
    label_map = json.loads(args.label_map)
    device = args.device or auto_device()
    skip = labeled_groups(Path(args.skip_labeled)) if args.skip_labeled else set()
    if skip:
        print(f"Bỏ qua {len(skip)} CV đã có trong project")

    model = CACHE.get(weights)
    version = CACHE.version(weights)
    groups = collect_pages(args.inputs, Path(args.pages_dir), args.dpi, skip)
    print(f"Model: {weights} ({version})  |  {len(groups)} CV\n")

    tasks = []
    for cv, pages in groups.items():
        units = [pages] if cfg.multipage else [[p] for p in pages]  # multi-page: 1 task / CV
        for unit in units:
            results, scores = [], []
            for k, page in enumerate(unit):
                img = open_rgb(page)
                dets = detect(model, img, args.conf, args.iou, args.imgsz, device)
                r, s = to_ls_results(dets, img.width, img.height, cfg,
                                     item_index=k if cfg.multipage else None, label_map=label_map)
                results += r
                scores += s
            urls = [local_files_url(p, Path(args.doc_root)) for p in unit]
            tasks.append({
                "data": {cfg.value_key: urls if cfg.multipage else urls[0], "cv_name": cv},
                "predictions": [{"model_version": version, "score": task_score(scores), "result": results}],
            })
            print(f"{cv:>10}: {len(unit)} trang, {len(results):3d} box, score {task_score(scores):.2f}")

    out = Path(args.out or LS_DIR / f"tasks_preannotated_{datetime.now():%Y%m%d_%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(tasks)} task -> {out}  (Project > Import)")


if __name__ == "__main__":
    main()
