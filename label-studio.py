"""
Khởi động Label Studio với local file serving trỏ vào BASE_DIR.

Trong Label Studio, tạo Local Storage với path:  <BASE_DIR>\\PNG
"""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, PNG_DIR

PORT = 8080


def find_label_studio():
    """Tìm executable trong cùng môi trường Python (Windows / Linux / macOS)."""
    bin_dir = Path(sys.executable).parent
    for cand in (bin_dir / "label-studio.exe", bin_dir / "Scripts" / "label-studio.exe",
                 bin_dir / "label-studio"):
        if cand.exists():
            return str(cand)
    return shutil.which("label-studio")


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    for path, what in ((BASE_DIR, "BASE_DIR"), (PNG_DIR, "thư mục ảnh")):
        if not path.is_dir():
            sys.exit(f"❌ Không tìm thấy {what}: {path}")

    exe = find_label_studio()
    if exe is None:
        sys.exit(f'❌ Chưa cài Label Studio. Cài bằng:\n"{sys.executable}" -m pip install label-studio')

    if port_in_use(PORT):
        sys.exit(f"❌ Port {PORT} đang được dùng (Label Studio đã chạy?). Mở http://localhost:{PORT}")

    # Chỉ đặt env cho process con, không sửa môi trường hiện tại
    env = {
        **os.environ,
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(BASE_DIR),
    }

    print("=" * 70)
    print(f"LABEL STUDIO  →  http://localhost:{PORT}")
    print(f"Document root : {BASE_DIR}")
    print(f"Local storage : {PNG_DIR}")
    print("Ctrl+C để dừng.")
    print("=" * 70)

    try:
        subprocess.run([exe, "start", "--port", str(PORT)], env=env, check=False)
    except KeyboardInterrupt:
        print("\nLabel Studio đã dừng.")


if __name__ == "__main__":
    main()
