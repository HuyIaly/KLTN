"""
Khởi động ML backend (YOLO gợi ý box) cho Label Studio.

Chạy SAU label-studio.py, ở cửa sổ terminal thứ 2:
    python ml-backend.py
    python ml-backend.py --python D:\\envs\\lsml\\Scripts\\python.exe   # nếu cài backend ở venv riêng

Trong Label Studio: Project > Settings > Model > Connect Model > http://localhost:9090
"""

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, ML_BACKEND_PORT, active_weights

ROOT = Path(__file__).resolve().parent
WSGI = ROOT / "labelstudio" / "ml_backend" / "_wsgi.py"


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python của môi trường đã cài label-studio-ml")
    ap.add_argument("--port", type=int, default=ML_BACKEND_PORT)
    args = ap.parse_args()

    weights = active_weights()
    if not weights.exists():
        sys.exit(f"❌ Không tìm thấy trọng số: {weights}\n   Train trước hoặc đặt YOLO_WEIGHTS.")

    if not Path(args.python).exists():
        sys.exit(f"❌ Không tìm thấy Python: {args.python}\n"
                 "   Tạo venv riêng cho backend:\n"
                 "     python -m venv D:\\envs\\lsml\n"
                 "     D:\\envs\\lsml\\Scripts\\python -m pip install -r labelstudio\\ml_backend\\requirements.txt")

    check = subprocess.run([args.python, "-c", "import label_studio_ml.api"], capture_output=True, check=False)
    if check.returncode != 0:
        sys.exit("❌ Môi trường chưa có label-studio-ml v2. Cài bằng:\n"
                 f'"{args.python}" -m pip install -r labelstudio\\ml_backend\\requirements.txt')

    if port_in_use(args.port):
        sys.exit(f"❌ Port {args.port} đang được dùng (backend đã chạy?).")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # log tiếng Việt trên console Windows

    print("=" * 70)
    print(f"ML BACKEND    →  http://localhost:{args.port}")
    print(f"Trọng số      : {weights}  (tự đổi sang weights\\finetune.pt khi retrain xong)")
    print(f"Đọc ảnh từ    : {BASE_DIR}")
    print("Label Studio  : Settings > Model > Connect Model > dán URL ở trên")
    print("Ctrl+C để dừng.")
    print("=" * 70)

    try:
        subprocess.run([args.python, str(WSGI), "--port", str(args.port)], env=env, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("\nML backend đã dừng.")


if __name__ == "__main__":
    main()
