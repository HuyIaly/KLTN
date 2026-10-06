"""Vòng lặp human-in-the-loop: export đã sửa -> YOLO data -> finetune -> ghi đè
FINETUNE_WEIGHTS -> ML backend tự nạp lại -> pre-annotate lô CV tiếp theo tốt hơn.

python labelstudio/retrain.py --export D:\\Final\\labelstudio\\export.json --epochs 60

Mỗi vòng có thư mục riêng BASE_DIR/ls_rounds/<thời gian>/ (raw + dataset) nên KHÔNG đụng
vào DATASET_DIR (dataset chính để báo cáo).
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import BEST_WEIGHTS, FINETUNE_WEIGHTS, LS_ROUNDS_DIR, PNG_DIR, WEIGHTS_DIR  # noqa: E402


def run(*cmd):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True, help="JSON export (định dạng JSON) từ Label Studio")
ap.add_argument("--images-dir", default=str(PNG_DIR))
ap.add_argument("--init", choices=["last", "base"], default="last",
                help="last: finetune tiếp từ model đang dùng (nhanh); base: từ <arch>.pt (sạch, để báo cáo)")
ap.add_argument("--arch", default="yolo11s")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default=None)
a = ap.parse_args()

stamp = datetime.now().strftime("%Y%m%d_%H%M")
raw, ds = LS_ROUNDS_DIR / stamp / "raw", LS_ROUNDS_DIR / stamp / "dataset"
run("training/labelstudio_to_yolo.py", "--export", a.export, "--images-dir", a.images_dir, "--out", raw)
run("training/split_dataset.py", "--src", raw, "--dst", ds)

current = FINETUNE_WEIGHTS if FINETUNE_WEIGHTS.exists() else BEST_WEIGHTS
init = current if a.init == "last" and current.exists() else f"{a.arch}.pt"
if FINETUNE_WEIGHTS.exists():  # giữ bản cũ để rollback
    shutil.copy2(FINETUNE_WEIGHTS, WEIGHTS_DIR / f"finetune_before_{stamp}.pt")

tmp = WEIGHTS_DIR / f"_tmp_{stamp}.pt"
cmd = ["training/train.py", "--mode", "finetune", "--data", ds / "data.yaml", "--arch", a.arch,
       "--weights", init, "--epochs", a.epochs, "--batch", a.batch, "--name", f"ls_round_{stamp}",
       "--save-as", tmp]
if a.device:
    cmd += ["--device", a.device]
run(*cmd)

tmp.replace(FINETUNE_WEIGHTS)                       # ghi đè nguyên tử ở bước cuối
tmp.with_suffix(".json").replace(FINETUNE_WEIGHTS.with_suffix(".json"))
print(f"\nKhởi tạo từ: {init}\nĐã cập nhật {FINETUNE_WEIGHTS}. ML backend dùng model mới ở request kế tiếp.")
