# Nối YOLO layout vào Label Studio (theo config.py, BASE_DIR = D:\Final)

```
D:\Final
├─ CV\            PDF gốc (CV_1.pdf, CV_2.pdf ...)         ← changename.py, check_duplicate.py
├─ PNG\           ảnh trang CV_1_page_001.png (300 DPI)   ← Local storage của Label Studio
├─ labelstudio\   label_config.xml, tasks_preannotated_*.json, export.json
├─ runs\          cv_layout_yolo\weights\best.pt (model hiện có), các run mới
├─ weights\       scratch.pt, finetune.pt (+ bản backup mỗi vòng retrain)
├─ ls_rounds\     dữ liệu từng vòng retrain
└─ YOLO_Dataset\  dataset chính để báo cáo
```

Mọi script lấy đường dẫn, `CLASS_NAMES`, `RENDER_DPI`, `IMGSZ` từ `config.py`. Đặt `config.py` ở gốc project (cùng chỗ `label-studio.py`). Muốn đổi thư mục gốc thì `set CV_BASE_DIR=E:\Other`.

## Lần đầu

```bat
:: 1. Label config sinh từ CLASS_NAMES -> D:\Final\labelstudio\label_config.xml
python labelstudio\make_config.py
:: 2. Chạy Label Studio (cửa sổ 1)
python label-studio.py
```
Trong Label Studio:

1. Dán nội dung `label_config.xml` vào **Settings → Labeling Interface → Code**.
2. Vào **Cloud Storage → Add Source Storage → Local files**, nhập path `D:\Final\PNG`. Không bật "Treat every bucket object as a source file", và không cần Sync.

```bat
:: 3. ML backend (cửa sổ 2). Nên cài label-studio-ml v2 trong venv riêng:
python -m venv D:\envs\lsml
D:\envs\lsml\Scripts\python -m pip install -r labelstudio\ml_backend\requirements.txt
python ml-backend.py --python D:\envs\lsml\Scripts\python.exe
```
Sau đó vào **Settings → Model → Connect Model**, nhập `http://localhost:9090`, rồi bật dùng predictions để prelabel.

Backend chọn trọng số mỗi request theo thứ tự ưu tiên sau:

1. Biến môi trường `YOLO_WEIGHTS`.
2. `D:\Final\weights\finetune.pt`.
3. `BEST_WEIGHTS`, tức model `runs\cv_layout_yolo` bạn đã train.

Vì vậy, khi `retrain.py` tạo xong `finetune.pt`, backend tự chuyển sang model mới mà không cần khởi động lại. Ảnh được đọc thẳng từ `D:\Final\PNG`, nên không cần API key.

## Mỗi lô CV mới

```bat
python changename.py --dry-run
python changename.py
python check_duplicate.py
:: Export project hiện tại (định dạng JSON) -> D:\Final\labelstudio\export.json, rồi:
python labelstudio\preannotate.py --skip-labeled D:\Final\labelstudio\export.json
```
Lệnh cuối làm bốn việc:

- Render những PDF chưa có ảnh trang. Trang đã có PNG thì giữ nguyên, nên không lệch với nhãn cũ.
- Chạy YOLO trên các trang.
- Bỏ qua các CV đã có trong project.
- Ghi kết quả ra `labelstudio\tasks_preannotated_<time>.json`.

Sau đó vào **Import** trong Label Studio, chọn file JSON đó. Mỗi CV là một task, có sẵn box gợi ý và trường `cv_name` để lọc.

Trong Data Manager, sắp xếp theo **Prediction score tăng dần** để sửa trước những CV mà model kém tự tin.

## Retrain sau khi sửa nhãn

```bat
:: Export JSON mới nhất rồi:
python labelstudio\retrain.py --export D:\Final\labelstudio\export.json --epochs 60
```
Quy trình chạy như sau:

1. Convert dữ liệu vào `ls_rounds\<time>\raw`.
2. Chia train/val/test theo CV (`config.cv_group`).
3. Finetune từ model đang dùng.
4. Ghi đè `weights\finetune.pt`. Bản cũ được backup thành `finetune_before_<time>.pt`.

`YOLO_Dataset` không bị đụng tới. Khi làm số liệu cho báo cáo, chạy lại từ đầu với `--init base` hoặc dùng các bước trong README chính:

```bat
python training\labelstudio_to_yolo.py --export D:\Final\labelstudio\export.json
python training\split_dataset.py --overwrite
python training\train.py --mode scratch
python training\train.py --mode finetune
python training\evaluate.py
```

## Lưu ý

- **Nhãn được submit.** Chỉ những task đã **Submit** mới có annotation trong file export; prediction không tự tính là nhãn.
- **Tên lớp.** Tên Label trong project phải đúng `CLASS_NAMES`. Nếu project cũ đặt tên khác, dùng `LABEL_MAP` trong `labelstudio\ml_backend\.env`.
- **Thứ tự lớp.** Đổi `CLASS_NAMES` (thêm hoặc đổi thứ tự) làm lệch id lớp của model cũ. Khi đó phải export, convert và train lại, không được dùng tiếp `best.pt` cũ.
- **Ghi rõ trong báo cáo.** Dữ liệu được gán nhãn có model hỗ trợ (model-assisted), và tập test nên được kiểm kỹ bằng tay.
