"""Chạy backend:  python labelstudio/ml_backend/_wsgi.py --port 9090
(Windows dùng server Flask này; Linux/Docker có thể dùng gunicorn _wsgi:app)"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # đọc .env cạnh file nếu có
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

from label_studio_ml.api import init_app  # noqa: E402
from model import CVLayoutBackend  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

app = init_app(model_class=CVLayoutBackend,
               basic_auth_user=os.getenv("BASIC_AUTH_USER"),
               basic_auth_pass=os.getenv("BASIC_AUTH_PASS"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=False)  # 1 luồng: tránh nạp YOLO song song
