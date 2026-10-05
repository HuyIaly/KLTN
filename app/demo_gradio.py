"""Demo Gradio: upload CV -> xem vùng layout, thứ tự đọc, text sau hậu xử lý, JSON.

python app/demo_gradio.py            (mở http://127.0.0.1:7860)
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from config import IMGSZ, RENDER_DPI, RUNS_DIR, WEIGHTS_DIR, auto_device  # noqa: E402
from cvparse.pipeline import CVLayoutPipeline, PipelineConfig  # noqa: E402

FILE_TYPES = [".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]


def find_weights() -> dict:
    found = {}
    for p in sorted(WEIGHTS_DIR.glob("*.pt")):          # D:\Final\weights\scratch.pt, finetune.pt
        if not p.stem.startswith("_tmp"):
            found[p.stem] = str(p)
    for p in sorted(RUNS_DIR.glob("**/weights/best.pt")):  # D:\Final\runs\<run>\weights\best.pt
        found[f"{p.parents[1].name}/best"] = str(p)
    return found


WEIGHTS = find_weights()


@lru_cache(maxsize=4)
def get_pipeline(weights_path: str, ocr_engine: str) -> CVLayoutPipeline:
    return CVLayoutPipeline(weights_path, device=auto_device(), imgsz=IMGSZ, ocr_engine=ocr_engine)


def _cfg(conf, iou, dpi, text_mode, order) -> PipelineConfig:
    return PipelineConfig(conf=float(conf), iou=float(iou), dpi=int(dpi),
                          text_mode=text_mode, order_strategy=order)


def run_single(file, model_name, conf, iou, dpi, text_mode, ocr_engine, order):
    if not file:
        raise gr.Error("Hãy upload một CV.")
    if model_name not in WEIGHTS:
        raise gr.Error(f"Chưa có trọng số trong {WEIGHTS_DIR} hoặc {RUNS_DIR}.")
    res = get_pipeline(WEIGHTS[model_name], ocr_engine).run(file, _cfg(conf, iou, dpi, text_mode, order))

    gallery = [(im, f"Trang {i + 1}") for i, im in enumerate(res.annotated_images())]
    data = res.to_dict()
    table = [[r["order"], r["page"] + 1, r["label"], round(r["score"], 3), r["source"],
              len(r["text"]), r["text"][:80].replace("\n", " ⏎ ")] for r in data["regions"]]
    tmp = Path(tempfile.mkdtemp()) / f"{Path(file).stem}_layout.json"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    t = res.timings
    stats = (f"**{len(res.pages)} trang · {len(res.sections)} mục · "
             f"{sum(r.label != 'UNASSIGNED' for r in res.regions)} vùng**  \n"
             f"Text layer: {', '.join('có' if p.has_text_layer else 'không' for p in res.pages)}  \n"
             f"Thời gian: chuẩn hoá {t['normalize']:.2f}s · YOLO {t['layout']:.2f}s · "
             f"lấy chữ {t['extract']:.2f}s · hậu xử lý {t['postprocess']:.2f}s · **tổng {t['total']:.2f}s**")
    return gallery, stats, res.to_markdown(), table, data, str(tmp)


def run_compare(file, model_a, model_b, conf, iou, dpi, text_mode, ocr_engine, order):
    if not file:
        raise gr.Error("Hãy upload một CV.")
    outs = []
    for name in (model_a, model_b):
        if name not in WEIGHTS:
            raise gr.Error(f"Không tìm thấy trọng số: {name}")
        res = get_pipeline(WEIGHTS[name], ocr_engine).run(file, _cfg(conf, iou, dpi, text_mode, order))
        imgs = [(im, f"{name} – trang {i + 1}") for i, im in enumerate(res.annotated_images())]
        regs = [r for r in res.regions if r.label != "UNASSIGNED"]
        unassigned = sum(len(r.words) for r in res.regions if r.label == "UNASSIGNED")
        info = (f"**{name}**: {len(regs)} vùng, {len(res.sections)} mục, "
                f"score TB {sum(r.score for r in regs) / max(1, len(regs)):.2f}, "
                f"{unassigned} từ không thuộc vùng nào, YOLO {res.timings['layout']:.2f}s")
        outs += [imgs, info]
    return outs


def build_ui():
    names = list(WEIGHTS) or ["(chưa có trọng số)"]
    default_a = "scratch" if "scratch" in WEIGHTS else names[0]
    default_b = "finetune" if "finetune" in WEIGHTS else names[-1]

    with gr.Blocks(title="CV Layout Pipeline") as demo:
        gr.Markdown("## CV Layout Pipeline\nChuẩn hoá → YOLO layout → Text layer / OCR → Hậu xử lý "
                    "(thứ tự đọc, nối trang). Đầu ra là JSON theo mục, sẵn sàng cho NER.")
        with gr.Row():
            file = gr.File(label="CV (PDF / DOCX / ảnh)", file_types=FILE_TYPES, type="filepath")
            with gr.Column():
                with gr.Row():
                    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Confidence")
                    iou = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="NMS IoU")
                    dpi = gr.Slider(100, 300, RENDER_DPI, step=25, label="DPI render",
                                    info="Nên giữ = RENDER_DPI lúc gán nhãn")
                with gr.Row():
                    text_mode = gr.Radio(["auto", "text_layer", "ocr"], value="auto", label="Nguồn chữ")
                    ocr_engine = gr.Radio(["easyocr", "tesseract"], value="easyocr", label="OCR")
                    order = gr.Radio(["x_first", "y_first"], value="x_first", label="Thứ tự đọc",
                                     info="x_first: hợp CV 2 cột")
        common = [conf, iou, dpi, text_mode, ocr_engine, order]

        with gr.Tab("Chạy pipeline"):
            with gr.Row():
                model = gr.Dropdown(names, value=default_b, label="Model")
                btn = gr.Button("Chạy", variant="primary")
            stats = gr.Markdown()
            with gr.Row():
                gallery = gr.Gallery(label="Vùng layout (#thứ tự đọc)", columns=2, height=720)
                text_md = gr.Markdown(label="Text theo mục", height=720)
            table = gr.Dataframe(headers=["#", "Trang", "Nhãn", "Score", "Nguồn", "Số ký tự", "Trích đoạn"],
                                 label="Các vùng theo thứ tự đọc", wrap=True)
            with gr.Row():
                js = gr.JSON(label="JSON (đầu vào cho NER)")
                dl = gr.File(label="Tải JSON")
            btn.click(run_single, [file, model] + common, [gallery, stats, text_md, table, js, dl])

        with gr.Tab("So sánh scratch vs finetune"):
            with gr.Row():
                ma = gr.Dropdown(names, value=default_a, label="Model A")
                mb = gr.Dropdown(names, value=default_b, label="Model B")
                btn2 = gr.Button("So sánh", variant="primary")
            with gr.Row():
                with gr.Column():
                    ia = gr.Markdown()
                    ga = gr.Gallery(columns=1, height=760)
                with gr.Column():
                    ib = gr.Markdown()
                    gb = gr.Gallery(columns=1, height=760)
            btn2.click(run_compare, [file, ma, mb] + common, [ga, ia, gb, ib])
    return demo


if __name__ == "__main__":
    build_ui().launch()
