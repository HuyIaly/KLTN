# CV Layout Pipeline (YOLO → Text layer/OCR → Hậu xử lý)

## Bắt đầu nhanh (giải nén thẳng vào D:\Final)

Giải nén toàn bộ zip vào `D:\Final` (cùng chỗ với thư mục `CV\`, `PNG\`, `YOLO_Dataset\`).
`config.py` trong zip là bản của bạn + vài hằng số mới ở cuối; `changename.py`, `check_duplicate.py`,
`label-studio.py` giữ nguyên.

```bat
cd /d D:\Final
.venv\Scripts\activate
pip install -r requirements.txt

:: --- Train ---
python training\labelstudio_to_yolo.py --export D:\Final\labelstudio\export.json
python training\split_dataset.py --overwrite
python training\train.py --mode finetune --batch 4
python training\download_doclaynet.py --file yolov11s-doclaynet.pt
python training\train.py --mode finetune --batch 4 --weights D:\Final\weights\pretrained\yolov11s-doclaynet.pt --name finetune_doclaynet_yolo11s --save-as D:\Final\weights\finetune_doclaynet.pt
python training\train.py --mode scratch --batch 4
python training\evaluate.py --models scratch=D:\Final\weights\scratch.pt finetune_coco=D:\Final\weights\finetune.pt finetune_doclaynet=D:\Final\weights\finetune_doclaynet.pt

:: --- Gán nhãn (terminal 1) ---
python labelstudio\make_config.py
python label-studio.py

:: --- ML backend (terminal 2; venv riêng, cần Git, chỉ tạo 1 lần) ---
python -m venv D:\envs\lsml
D:\envs\lsml\Scripts\python -m pip install -r labelstudio\ml_backend\requirements.txt
python ml-backend.py --python D:\envs\lsml\Scripts\python.exe

:: --- Demo ---
python app\demo_gradio.py
```
Chi tiết Label Studio: `labelstudio\README.md`.

---

```
CV (PDF/DOCX/ảnh) → Chuẩn hoá → YOLO layout → [Text layer | OCR] → Hậu xử lý → JSON theo mục
   normalize.py      layout.py    extract.py / ocr.py               postprocess.py   pipeline.py
```
Dừng ở bước hậu xử lý: trường `sections` trong JSON là đầu vào cho NER (PhoBERT-CRF) chạy theo từng vùng.

## Cấu trúc

```
cvparse/          normalize, layout, ocr, extract, postprocess, visualize, pipeline, cli
training/         labelstudio_to_yolo.py, split_dataset.py, train.py, evaluate.py
app/demo_gradio.py
config.py         MỌI đường dẫn (BASE_DIR = D:\Final), CLASS_NAMES, RENDER_DPI, IMGSZ
label-studio.py   chạy Label Studio (document root = BASE_DIR)
ml-backend.py     chạy ML backend YOLO cho Label Studio
changename.py, check_duplicate.py   chuẩn bị PDF
```

## Cài đặt (Windows, venv)

```bat
python -m venv .venv && .venv\Scripts\activate
:: Có GPU: cài torch CUDA trước theo pytorch.org, ví dụ
:: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
DOCX cần LibreOffice (hoặc `pip install docx2pdf` + MS Word). Tesseract là tuỳ chọn.

## 1. Chuẩn bị dữ liệu từ Label Studio

Export project dạng **JSON** (không phải JSON-MIN), rồi:

```bat
python training\labelstudio_to_yolo.py --export D:\Final\labelstudio\export.json
python training\split_dataset.py --overwrite
:: mặc định: ảnh từ D:\Final\PNG -> D:\Final\ls_rounds\raw -> D:\Final\YOLO_Dataset
```
Chia theo CV (1 task = 1 CV) nên các trang của cùng một CV không bị rò rỉ giữa train và test.

## 2. Train từ đầu và finetune

```bat
python training\train.py --mode scratch  --arch yolo11s --epochs 300
python training\train.py --mode finetune --arch yolo11s --epochs 100
```
| | scratch | finetune |
|---|---|---|
| Khởi tạo | `yolo11s.yaml`, trọng số ngẫu nhiên | `yolo11s.pt` (COCO) hoặc `--weights` của bạn |
| Optimizer / lr0 | SGD / 0.01 | AdamW / 0.001 |
| Epoch / patience | 300 / 100 | 100 / 30 |
| Tuỳ chọn | | `--freeze 10` đóng băng backbone |

Augmentation chung trong `DOC_AUG`: không lật, không xoay, mosaic nhẹ.

Hai lựa chọn mở rộng:

- **Hai giai đoạn:** `--mode scratch --data doclaynet.yaml`, sau đó `--mode finetune --weights D:\Final\runs\<run>\weights\best.pt` (data mặc định là YOLO_Dataset).
- **Muốn so sánh công bằng:** giữ nguyên `--arch`, `--imgsz` và tập chia dữ liệu giữa hai chế độ.

## 3. Đánh giá

```bat
python training\evaluate.py
:: thêm model cũ: --models scratch=D:\Final\weights\scratch.pt finetune=D:\Final\weights\finetune.pt old=D:\Final\runs\cv_layout_yolo\weights\best.pt
```
Lệnh này in bảng mAP50 / mAP50-95 / ms/img và AP theo lớp, đồng thời ghi CSV vào `runs\eval`.

## 4. Chạy pipeline

```bat
:: weights/DPI/IMGSZ mặc định từ config.py, kết quả ra D:\Final\outputs
python -m cvparse.cli D:\Final\CV\CV_1.pdf
python app\demo_gradio.py
```
Demo có 2 tab: **Chạy pipeline** (ảnh vùng + thứ tự đọc, text theo mục, bảng vùng, JSON tải về) và **So sánh scratch vs finetune** (hai model chạy trên cùng một CV, hiển thị cạnh nhau).

## Hậu xử lý làm gì

1. **Lấy chữ:** Trang có text layer thì lấy word từ PDF, có xử lý trang bị xoay. Trang scan hoặc ảnh thì OCR cả trang. Mỗi word được gán cho đúng 1 vùng, ưu tiên vùng nhỏ hơn khi các vùng lồng nhau. Nếu vùng rỗng trên trang có text layer thì OCR riêng vùng đó.
2. **Snap box:** Box YOLO được bó sát theo chữ thực tế. Box gốc vẫn được lưu ở trường `det_bbox`.
3. **Dựng dòng và làm sạch:** Chuẩn hoá Unicode NFC (quan trọng với tiếng Việt), chuẩn hoá bullet, bỏ ký tự ẩn và ligature, nối từ bị gạch nối cuối dòng.
4. **Thứ tự đọc:** Dùng XY-cut đệ quy. Chế độ `x_first` (mặc định) gộp các dải 2 cột liền kề để đọc hết cột trái rồi mới sang cột phải. Chế độ `y_first` là XY-cut kinh điển.
5. **Header/footer:** Bỏ số trang và các dòng lặp lại ở lề trên/dưới của nhiều trang.
6. **Nối vùng và nối trang:** Các vùng cùng nhãn liền kề trên cùng một trang được gộp lại. Vùng cuối trang p và vùng đầu trang p+1 nếu cùng nhãn cũng được gộp thành 1 mục.
7. **Chữ ngoài vùng:** Chữ không thuộc vùng nào được giữ lại trong vùng `UNASSIGNED`, giúp phát hiện các mục YOLO bỏ sót.

## Định dạng JSON

```json
{
  "sections": [{"id": 3, "label": "EXPERIENCE", "pages": [0, 1], "source": "text_layer",
                "region_orders": [3, 4, 5], "text": "Kinh nghiệm làm việc\nCông ty ABC ..."}],
  "regions":  [{"order": 0, "page": 0, "label": "PROFILE", "score": 0.93, "bbox": [...],
                "det_bbox": [...], "source": "text_layer", "text": "...", "lines": [...]}],
  "pages": [...], "timings_sec": {...}, "config": {...}
}
```
