"""cvparse – pipeline phân tích bố cục CV: chuẩn hoá -> YOLO layout -> text layer/OCR -> hậu xử lý."""
from .schema import Word, Line, Page, Region, Section
from .pipeline import CVLayoutPipeline, PipelineConfig, PipelineResult

__all__ = ["Word", "Line", "Page", "Region", "Section",
           "CVLayoutPipeline", "PipelineConfig", "PipelineResult"]
