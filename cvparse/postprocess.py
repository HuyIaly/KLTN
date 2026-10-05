"""Bước 4 – Hậu xử lý: dựng dòng, làm sạch chữ, thứ tự đọc, bỏ header/footer, nối trang."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence

from .geometry import expand, union
from .schema import UNASSIGNED, BBox, Line, Region, Section, Word

# ----------------------------------------------------------------------------- làm sạch chữ
_BULLETS = "•●▪■□◦○‣⁃∙·➢➤►▶✓✔❖◆◇★☆*"
_BULLET_RE = re.compile(rf"^\s*[{re.escape(_BULLETS)}]+\s*")
_DASH_BULLET_RE = re.compile(r"^\s*[-–—]\s+")
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"})


def clean_line(text: str) -> str:
    # NFC rất quan trọng với tiếng Việt: PDF hay lưu dấu dạng tổ hợp (a + ̀ ) -> NER/tokenizer lệch.
    s = unicodedata.normalize("NFC", text).translate(_INVISIBLE).translate(_LIGATURES)
    s = re.sub(r"\s+", " ", s).strip()
    s = _BULLET_RE.sub("- ", s)
    s = _DASH_BULLET_RE.sub("- ", s)
    return s.strip()


def join_lines(lines: Sequence[str]) -> str:
    text = "\n".join(l for l in lines if l)
    # nối từ bị gạch nối cuối dòng (chủ yếu CV tiếng Anh): "develop-\nment" -> "development"
    return re.sub(r"(\w)-\n([a-zà-ỹ])", r"\1\2", text)


# ----------------------------------------------------------------------------- word -> dòng
def build_lines(words: List[Word], y_overlap: float = 0.5) -> List[Line]:
    if not words:
        return []
    ws = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    groups: List[dict] = []
    for w in ws:
        h = w.bbox[3] - w.bbox[1]
        target = None
        for g in reversed(groups[-4:]):
            gb = g["bbox"]
            ov = min(gb[3], w.bbox[3]) - max(gb[1], w.bbox[1])
            if ov > 0 and ov >= y_overlap * min(h, gb[3] - gb[1]):
                target = g
                break
        if target is None:
            groups.append({"words": [w], "bbox": w.bbox})
        else:
            target["words"].append(w)
            target["bbox"] = union([target["bbox"], w.bbox])

    lines: List[Line] = []
    for g in groups:
        g_ws = sorted(g["words"], key=lambda w: w.bbox[0])
        text = clean_line(" ".join(w.text for w in g_ws))
        if text in ("", "-"):
            continue
        conf = sum(w.conf for w in g_ws) / len(g_ws)
        lines.append(Line(text, g["bbox"], conf))
    lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
    return lines


def snap_bbox(region: Region, pad: float, w: int, h: int) -> BBox:
    """Bó box YOLO sát theo chữ thực tế (OCR/text layer) -> crop gọn hơn cho bước sau."""
    if not region.words:
        return region.bbox
    return expand(union(x.bbox for x in region.words), pad, w, h)


# ----------------------------------------------------------------------------- thứ tự đọc
def reading_order(boxes: List[BBox], page_w: float, page_h: float, strategy: str = "x_first",
                  gap_frac: float = 0.005, shrink: float = 0.03) -> List[int]:
    """Recursive XY-cut trên các vùng YOLO.

    x_first: cắt cột trước (hợp CV 2 cột: đọc hết cột trái rồi cột phải).
    y_first: cắt hàng trước (XY-cut kinh điển; hợp CV 1 cột có hàng ngang nhiều box).
    `shrink` thu nhỏ mỗi box khi chiếu để box YOLO lấn nhẹ qua khe cột không chặn phép cắt.
    """
    if not boxes:
        return []
    gaps = (gap_frac * page_w, gap_frac * page_h)
    axes = (0, 1) if strategy == "x_first" else (1, 0)

    def split(idxs: List[int], axis: int) -> List[List[int]]:
        ivs = []
        for i in idxs:
            b = boxes[i]
            lo, hi = (b[0], b[2]) if axis == 0 else (b[1], b[3])
            s = (hi - lo) * shrink
            ivs.append((lo + s, hi - s, i))
        ivs.sort()
        groups, cur, cur_hi = [], [ivs[0][2]], ivs[0][1]
        for lo, hi, i in ivs[1:]:
            if lo > cur_hi + gaps[axis]:
                groups.append(cur)
                cur, cur_hi = [i], hi
            else:
                cur.append(i)
                cur_hi = max(cur_hi, hi)
        groups.append(cur)
        return groups

    def merge_column_bands(groups: List[List[int]]) -> List[List[int]]:
        # Cắt ngang có thể chặt 1 khối 2 cột thành nhiều dải (vì khe giữa các mục 2 cột tình cờ
        # thẳng hàng). Gộp lại các dải LIỀN KỀ đều có nhiều cột để đọc hết cột trái rồi cột phải;
        # dải 1 cột (header, mục full-width) vẫn là ranh giới.
        merged: List[List[int]] = []
        prev_multi = False
        for g in groups:
            multi = len(g) > 1 and len(split(g, 0)) > 1
            if merged and multi and prev_multi:
                merged[-1] = merged[-1] + g
            else:
                merged.append(list(g))
            prev_multi = multi
        return merged if len(merged) > 1 else groups

    def rec(idxs: List[int], depth: int = 0) -> List[int]:
        if len(idxs) <= 1 or depth > 64:
            return sorted(idxs, key=lambda i: (boxes[i][1], boxes[i][0]))
        for axis in axes:
            groups = split(idxs, axis)
            if len(groups) > 1:
                if axis == 1 and strategy == "x_first":
                    groups = merge_column_bands(groups)
                return [i for g in groups for i in rec(g, depth + 1)]
        return sorted(idxs, key=lambda i: (boxes[i][1], boxes[i][0]))

    return rec(list(range(len(boxes))))


# ----------------------------------------------------------------------------- header / footer
_PAGE_NUM_RE = re.compile(r"^(page|trang)?\s*\d{1,3}\s*((/|of|trên|\|)\s*\d{1,3})?$", re.I)


def _hf_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def strip_headers_footers(regions: List[Region], page_heights: Dict[int, int],
                          margin: float = 0.07) -> int:
    """Bỏ số trang và dòng lặp lại ở lề trên/dưới của nhiều trang. Trả về số dòng đã bỏ."""
    n_pages = len(page_heights)

    def in_margin(line: Line, page: int) -> bool:
        h = page_heights[page]
        return line.bbox[3] <= margin * h or line.bbox[1] >= (1 - margin) * h

    counts: Counter = Counter()
    for p in page_heights:
        keys = {_hf_key(l.text) for r in regions if r.page == p for l in r.lines if in_margin(l, p)}
        counts.update(keys)
    repeated = {k for k, c in counts.items() if n_pages >= 2 and c >= max(2, math.ceil(0.6 * n_pages))}

    removed = 0
    for r in regions:
        keep = []
        for l in r.lines:
            if in_margin(l, r.page) and (_hf_key(l.text) in repeated or _PAGE_NUM_RE.match(l.text)):
                removed += 1
            else:
                keep.append(l)
        r.lines = keep
    return removed


# ----------------------------------------------------------------------------- nối vùng / nối trang
def merge_sections(ordered: List[Region], merge_same_page: bool = True,
                   merge_cross_page: bool = True) -> List[Section]:
    """`ordered` = vùng của mọi trang theo thứ tự đọc (không gồm UNASSIGNED).

    - Cùng trang, liền kề, cùng nhãn  -> 1 mục (YOLO tách 1 mục thành nhiều box).
    - Vùng CUỐI trang p và vùng ĐẦU trang p+1 cùng nhãn -> 1 mục (mục bị ngắt trang).
    """
    sections: List[Section] = []
    for r in ordered:
        if r.label == UNASSIGNED or not r.lines:
            if r.label != UNASSIGNED and not r.lines:
                sections.append(Section(len(sections), r.label, [r]))  # vùng rỗng vẫn giữ (vd PHOTO)
            continue
        prev = sections[-1] if sections else None
        if prev is not None and prev.label == r.label and prev.parts[-1].lines:
            last = prev.parts[-1]
            if (merge_same_page and last.page == r.page) or \
               (merge_cross_page and r.page == last.page + 1):
                prev.parts.append(r)
                continue
        sections.append(Section(len(sections), r.label, [r]))
    return sections
