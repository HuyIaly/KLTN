"""Tải trọng số YOLO đã train trên DocLayNet (hantian/yolo-doclaynet) để làm điểm khởi tạo finetune.

pip install huggingface_hub
python training/download_doclaynet.py --list                         # xem các file có trên repo
python training/download_doclaynet.py --file yolov11s-doclaynet.pt   # tải về WEIGHTS_DIR/pretrained
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WEIGHTS_DIR  # noqa: E402

REPO = "hantian/yolo-doclaynet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--file", default="yolov11s-doclaynet.pt")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "pretrained"))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    if args.list:
        for f in sorted(list_repo_files(REPO)):
            if f.endswith(".pt"):
                print(f)
        return

    path = Path(hf_hub_download(REPO, args.file, local_dir=args.out))
    print(f"Đã tải: {path}")

    # Kiểm tra file nạp được bằng Ultralytics đang cài + in thông tin để ghi vào báo cáo
    from ultralytics import YOLO
    m = YOLO(str(path))
    yaml_name = getattr(m.model, "yaml", {}).get("yaml_file", "?") if hasattr(m.model, "yaml") else "?"
    print(f"Kiến trúc : {yaml_name}")
    print(f"Lớp gốc   : {len(m.names)} lớp DocLayNet -> {list(m.names.values())}")
    print("(Head phân lớp sẽ được khởi tạo lại cho CLASS_NAMES khi finetune, backbone + neck giữ nguyên.)")


if __name__ == "__main__":
    main()
