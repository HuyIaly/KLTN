"""Direct PDF/DOCX branch: section detection from the document structure, no rendering, no YOLO.

Lines come from page.get_text("dict") with font size / bold / bbox. A line is a section
heading when it looks like one (short, larger/bold/UPPER) AND matches the bilingual
keyword dictionary in sections.yaml (exact or fuzzy, Ngo, Two-Phase CV).
In a 2-column page the right column restarts as HEADER until its first heading, because
the name / job title usually sits there above the sections.
DOCX: paragraphs of word/document.xml (tables and text boxes included, in document order)
plus page headers, with size/bold resolved from runs and styles.xml. Stdlib only.
HEADER text (before the first heading) is split line by line into the YOLO classes
NAME / JOB_TITLE / CONTACT / SUMMARY, so every section_label is one of config.CLASS_NAMES.

Output follows CLAUDE.md:
    [{section_label, heading_text, text, source: "pdf"|"docx", page, bbox}]
    bbox in PDF points; None for DOCX (no layout without rendering)

CLI:
    python -m cvparse.pdf_layout CV --out out/pdf_sections.jsonl --ls-tasks out/ner_tasks.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz

from .normalize import page_has_text_layer
from .postprocess import clean_line
from .regex_fields import find_fields

DEFAULT_CONFIG = Path(__file__).with_name("sections.yaml")
HEADER = "HEADER"
CONNECTORS = {"va", "and", "cua", "toi", "my"}  # text before the first heading: name, job title, contact
BOLD_FLAG = 16


@dataclass
class PdfLine:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    page: int
    column: int = 0  # 0 single/full-width, 1 left, 2 right (set by reading_order)


def fold(text: str) -> str:
    """'Học Vấn & Chứng chỉ' -> 'hoc van chung chi' (for keyword matching only)."""
    s = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["keywords"] = [(fold(k), label) for label, kws in cfg["labels"].items() for k in kws]
    return cfg


# ----------------------------------------------------------------------------- lines
def read_lines(page: fitz.Page, page_index: int) -> list[PdfLine]:
    lines, seen = [], set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = clean_line(" ".join(s["text"] for s in spans))
            # Canva often draws the same text twice (shadow / fake bold)
            key = (text, round(line["bbox"][0] / 3), round(line["bbox"][1] / 3))
            if not text or key in seen:
                continue
            seen.add(key)
            bold = all(s["flags"] & BOLD_FLAG or "bold" in s["font"].lower() for s in spans)
            size = max(s["size"] for s in spans)
            lines.append(PdfLine(text, tuple(line["bbox"]), size, bold, page_index))
    return merge_same_row(lines)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _docx_styles(z: zipfile.ZipFile) -> dict[str, tuple[float, bool, bool]]:
    """styleId -> (size pt, bold, is_heading), following basedOn."""
    if "word/styles.xml" not in z.namelist():
        return {}
    raw = {}
    for st in ET.fromstring(z.read("word/styles.xml")).iter(f"{W}style"):
        sid = st.get(f"{W}styleId")
        sz, b = st.find(f"{W}rPr/{W}sz"), st.find(f"{W}rPr/{W}b")
        based = st.find(f"{W}basedOn")
        name = (st.find(f"{W}name").get(f"{W}val") if st.find(f"{W}name") is not None else sid) or ""
        raw[sid] = (float(sz.get(f"{W}val")) / 2 if sz is not None else None,
                    None if b is None else b.get(f"{W}val", "1") not in ("0", "false", "off"),
                    name.lower().startswith(("heading", "title")),
                    based.get(f"{W}val") if based is not None else None)

    def resolve(sid, depth=0):
        size, bold, head, base = raw.get(sid, (None, None, False, None))
        if base and depth < 10 and (size is None or bold is None):
            bs, bb, bh = resolve(base, depth + 1)
            size, bold, head = size or bs, bold if bold is not None else bb, head or bh
        return size or 0.0, bool(bold), head
    return {sid: resolve(sid) for sid in raw}


def read_docx_lines(path: str | Path) -> list[PdfLine]:
    """One PdfLine per non-empty paragraph: page headers first, then the body in document order."""
    lines: list[PdfLine] = []
    with zipfile.ZipFile(path) as z:
        styles = _docx_styles(z)
        parts = sorted(n for n in z.namelist() if re.fullmatch(r"word/header\d*\.xml", n))
        for part in parts + ["word/document.xml"]:
            root = ET.fromstring(z.read(part))
            skip = {id(p) for fb in root.iter(MC_FALLBACK) for p in fb.iter(f"{W}p")}  # VML copy of text boxes
            for p in root.iter(f"{W}p"):
                if id(p) in skip or p.find(f".//{W}txbxContent") is not None:
                    continue  # a paragraph that hosts a text box: its own text comes via the box
                ps = p.find(f"{W}pPr/{W}pStyle")
                psize, pbold, head = styles.get(ps.get(f"{W}val") if ps is not None else "Normal", (0.0, False, False))
                parts_txt, sizes, bolds = [], [], []
                for r in p.iter(f"{W}r"):
                    t = "".join(x.text or "" for x in r.iter() if x.tag in (f"{W}t", f"{W}tab"))
                    if r.find(f"{W}tab") is not None and not t:
                        t = " "
                    if not t.strip():
                        parts_txt.append(t)
                        continue
                    sz, b = r.find(f"{W}rPr/{W}sz"), r.find(f"{W}rPr/{W}b")
                    sizes.append(float(sz.get(f"{W}val")) / 2 if sz is not None else psize)
                    bolds.append(b.get(f"{W}val", "1") not in ("0", "false", "off") if b is not None else pbold)
                    parts_txt.append(t)
                text = clean_line("".join(parts_txt))
                if text:
                    lines.append(PdfLine(text, (0.0, 0.0, 0.0, 0.0), max(sizes, default=psize) + (2 if head else 0),
                                         head or (bool(bolds) and all(bolds)), 0))
    return lines


def merge_same_row(lines: list[PdfLine], max_gap: float = 6.0) -> list[PdfLine]:
    """PDFs often split one visual line into many tiny lines; glue horizontal neighbours."""
    out: list[PdfLine] = []
    for ln in sorted(lines, key=lambda line: (round(line.bbox[1]), line.bbox[0])):
        prev = out[-1] if out else None
        if (prev and abs(prev.bbox[1] - ln.bbox[1]) < 2 and abs(prev.bbox[3] - ln.bbox[3]) < 2
                and 0 <= ln.bbox[0] - prev.bbox[2] < max_gap):
            out[-1] = PdfLine(f"{prev.text} {ln.text}",
                              (prev.bbox[0], min(prev.bbox[1], ln.bbox[1]), ln.bbox[2],
                               max(prev.bbox[3], ln.bbox[3])),
                              max(prev.size, ln.size), prev.bold and ln.bold, ln.page)
        else:
            out.append(ln)
    return out


def column_split(lines: list[PdfLine], width: float, min_gap: float) -> float | None:
    """x of a vertical gutter crossed by the fewest lines (2-column CVs), or None."""
    lo, hi = int(width * min_gap), int(width * (1 - min_gap))
    best_x, best_cross = None, len(lines)
    for x in range(lo, hi):
        cross = sum(line.bbox[0] < x < line.bbox[2] for line in lines)
        if cross < best_cross:
            best_x, best_cross = x, cross
    if best_x is None:
        return None
    left = sum(line.bbox[2] <= best_x for line in lines)
    right = sum(line.bbox[0] >= best_x for line in lines)
    # ponytail: accept a split only if few lines cross it; per-band splits if CVs mix layouts
    if min(left, right) >= 3 and best_cross <= 0.15 * len(lines):
        return float(best_x)
    return None


def reading_order(lines: list[PdfLine], width: float, min_gap: float) -> list[PdfLine]:
    """Full-width lines cut the page into bands; inside a band: left column, then right."""
    split = column_split(lines, width, min_gap)
    lines = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
    if split is None:
        return lines
    out, left, right = [], [], []
    for ln in lines:
        if ln.bbox[0] < split < ln.bbox[2]:  # full-width line closes the band
            out += left + right + [ln]
            left, right = [], []
        elif ln.bbox[2] <= split:
            ln.column = 1
            left.append(ln)
        else:
            ln.column = 2
            right.append(ln)
    return out + left + right


# ----------------------------------------------------------------------------- headings
def letter_spaced(text: str) -> bool:
    """'K I N H  N G H I Ệ M' style headings."""
    toks = text.split()
    return len(toks) > 2 and sum(len(w) <= 2 for w in toks) > len(toks) / 2


def match_label(text: str, keywords: list[tuple[str, str]], fuzzy_ratio: float) -> str | None:
    t = fold(text)
    if not t:
        return None
    if letter_spaced(t):  # glue it back and compare without spaces
        compact = t.replace(" ", "")
        for k, label in keywords:
            if k.replace(" ", "") == compact:
                return label
    # Combined heading: label of the keyword that starts earliest, longest keyword on ties.
    hits = [(f" {t} ".find(f" {k} "), -len(k), k, label) for k, label in keywords if f" {k} " in f" {t} "]
    if hits:
        covered = set().union(*(k.split() for _, _, k, _ in hits)) | CONNECTORS
        # "Kỹ năng liên quan" ok (starts with keyword); "Cao đẳng Ngoại ngữ" is a degree, not LANGUAGE
        starts = min(hits)[0] == 0 and len(t.split()) <= len(min(hits)[2].split()) + 2
        if starts or set(t.split()) <= covered:
            return min(hits)[3]
        return None
    best = max(keywords, key=lambda kw: difflib.SequenceMatcher(None, t, kw[0]).ratio())
    return best[1] if difflib.SequenceMatcher(None, t, best[0]).ratio() >= fuzzy_ratio else None


# ----------------------------------------------------------------------------- header split
def name_key(lines: list[PdfLine], headings: dict[int, str]) -> str:
    """Folded text of the CV owner's name: the largest short, digit-free line on the first page."""
    cands = [ln for i, ln in enumerate(lines)
             if i not in headings and ln.page == lines[0].page and len(ln.text.split()) <= 6
             and not any(c.isdigit() for c in ln.text) and not find_fields(ln.text)]
    return fold(max(cands, key=lambda ln: ln.size).text) if cands else ""


def header_label(ln: PdfLine, prev: str, prev_ln: PdfLine | None, name: str, cfg: dict) -> str:
    """YOLO class of one line that sits before any section heading."""
    h, t = cfg["header"], fold(ln.text)
    if t and t == name:
        return "NAME"
    if (any(f[0] != "DATE" for f in find_fields(ln.text)) or "@" in ln.text
            or any(t.startswith(k) for k in h["contact_keywords"])):
        return "CONTACT"
    if prev in ("NAME", "JOB_TITLE") and (round(ln.size), ln.bold) == (round(prev_ln.size), prev_ln.bold):
        return prev  # name / job title wrapped onto a second line
    words = len(ln.text.split())
    if prev == "NAME" and words <= h["job_title_max_words"]:
        return "JOB_TITLE"
    if prev == "SUMMARY" or (words >= h["summary_min_words"] and not any(c.isdigit() for c in ln.text)):
        return "SUMMARY"
    # ponytail: any other short header line (city, birthday without keyword) counts as CONTACT
    return "CONTACT"


def looks_like_heading(ln: PdfLine, body_size: float, cfg: dict) -> bool:
    h = cfg["heading"]
    words = ln.text.split()
    n_words = 1 + ln.text.count("  ") if letter_spaced(ln.text) else len(words)
    if not words or n_words > h["max_words"] or ln.text.rstrip().endswith((".", ",", ";")):
        return False
    letters = [c for c in ln.text if c.isalpha()]
    return bool(letters) and (ln.size > body_size + h["size_margin"] or ln.bold or ln.text.isupper())


def style(ln: PdfLine) -> tuple[int, bool, bool]:
    return round(ln.size), ln.bold, ln.text.isupper()


def find_headings(lines: list[PdfLine], cfg: dict) -> dict[int, str]:
    """index in `lines` -> section label."""
    sizes = Counter()
    for ln in lines:
        sizes[round(ln.size, 1)] += len(ln.text)
    body = sizes.most_common(1)[0][0] if sizes else 0.0
    cands = [i for i, ln in enumerate(lines) if looks_like_heading(ln, body, cfg)]
    found = {i: lab for i in cands
             if (lab := match_label(lines[i].text, cfg["keywords"], cfg["heading"]["fuzzy_ratio"]))}
    # Headings of one CV usually share a style. When they do, a partial match with a one-off
    # style ("NGÔN NGỮ ANH" as a degree line) is dropped; an exact keyword is always kept.
    styles = Counter(style(lines[i]) for i in found)
    shared = max(styles.values(), default=0) >= 2
    exact = {k for k, _ in cfg["keywords"]}
    # ponytail: unknown headings stay inside the previous section; style propagation to an
    # OTHER label was tried and lowered val accuracy (false headings)
    kept = {i: lab for i, lab in found.items()
            if not shared or styles[style(lines[i])] >= 2 or fold(lines[i].text) in exact}
    if shared:
        # long heading in the shared style that starts with a keyword ("KINH NGHIỆM VÀ CHƯƠNG TRÌNH")
        head_style = styles.most_common(1)[0][0]
        for i in cands:
            t = f"{fold(lines[i].text)} "
            if i not in kept and style(lines[i]) == head_style:
                kept |= {i: lab for k, lab in cfg["keywords"] if t.startswith(f"{k} ")}
    # a heading text repeated in one CV is a sub-heading ("THÀNH TÍCH" under each job): drop it
    # ponytail: also drops a real heading repeated on page 2; keep the first one if that shows up
    repeats = Counter(fold(lines[i].text) for i in kept)
    return {i: lab for i, lab in kept.items() if repeats[fold(lines[i].text)] == 1}


# ----------------------------------------------------------------------------- sections
def _union(lines: list[PdfLine]) -> list[float]:
    return [round(min(line.bbox[0] for line in lines), 1), round(min(line.bbox[1] for line in lines), 1),
            round(max(line.bbox[2] for line in lines), 1), round(max(line.bbox[3] for line in lines), 1)]


def read_pdf_lines(path: str | Path, cfg: dict) -> list[PdfLine]:
    """Pages without a text layer are skipped (they go to YOLO+OCR)."""
    lines: list[PdfLine] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            if page_has_text_layer(page):
                lines += reading_order(read_lines(page, i), page.rect.width,
                                       cfg["heading"]["column_min_gap"])
    return lines


def is_contact_only(text: str) -> bool:
    fields = [f for f in find_fields(text) if f[0] != "DATE"]
    return bool(fields) and sum(len(f[3]) for f in fields) >= 0.8 * len(text.replace(" ", ""))


def side_headings(lines: list[PdfLine], headings: dict[int, str]) -> tuple[list[PdfLine], dict[int, str]]:
    """Pages whose left column holds (mostly) headings only: headings sit beside their content,
    so read the page row by row instead of column by column."""
    out, new_heads = [], {}
    for page in sorted({ln.page for ln in lines}):
        idx = [i for i, ln in enumerate(lines) if ln.page == page]
        left = [i for i in idx if lines[i].column == 1]
        n_head = sum(i in headings for i in left)
        if n_head >= 2 and n_head >= 0.4 * len(left):
            # heading continuation lines ("KINH NGHIỆM" / "LÀM VIỆC") are lifted with their heading
            idx.sort(key=lambda i: (lines[i].bbox[1] - (6 if lines[i].column == 1 else 0), lines[i].bbox[0]))
            for i in idx:
                lines[i].column = 0
        for i in idx:
            if i in headings:
                new_heads[len(out)] = headings[i]
            out.append(lines[i])
    return out, new_heads


def parse_document(path: str | Path, cfg: dict | None = None) -> list[dict]:
    """Sections of one PDF or DOCX, labelled with the YOLO class names."""
    cfg = cfg or load_config()
    source = "docx" if Path(path).suffix.lower() == ".docx" else "pdf"
    lines = read_docx_lines(path) if source == "docx" else read_pdf_lines(path, cfg)
    if not lines:
        return []
    headings = find_headings(lines, cfg)
    if source == "pdf":
        lines, headings = side_headings(lines, headings)
    name = name_key(lines, headings)

    # right column restarts as HEADER only above the page's first heading (name / title area)
    top_heading: dict[int, float] = {}
    for i in headings:
        top_heading[lines[i].page] = min(top_heading.get(lines[i].page, float("inf")), lines[i].bbox[1])

    # label every body line; heading lines only switch the label
    tagged: list[tuple[str, str, PdfLine]] = []
    label, heading, prev, prev_ln = HEADER, "", "", None
    for i, ln in enumerate(lines):
        if (ln.column == 2 and (i == 0 or lines[i - 1].column != 2) and i not in headings
                and ln.bbox[1] < top_heading.get(ln.page, float("inf"))):
            label, heading, prev = HEADER, "", ""
        if i in headings:
            label, heading = headings[i], ln.text
            continue
        lab = header_label(ln, prev, prev_ln, name, cfg) if label == HEADER else label
        if lab not in ("CONTACT", "REFERENCE") and is_contact_only(ln.text):
            lab = "CONTACT"  # e.g. an email in a footer strip under EXPERIENCE
        prev, prev_ln = lab, ln
        tagged.append((lab, heading if label != HEADER else "", ln))

    # consecutive lines with the same label/heading/page form one entry
    sections: list[dict] = []
    for lab, head, ln in tagged:
        last = sections[-1] if sections else None
        if last and (last["section_label"], last["heading_text"], last["page"]) == (lab, head, ln.page):
            last["_lines"].append(ln)
        else:
            sections.append({"section_label": lab, "heading_text": head, "source": source,
                             "page": ln.page, "_lines": [ln]})
    for sec in sections:
        part = sec.pop("_lines")
        sec["text"] = "\n".join(line.text for line in part)
        sec["bbox"] = _union(part) if source == "pdf" else None
    return sections


parse_pdf = parse_document  # old name


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="PDF/DOCX file or folder")
    ap.add_argument("--out", type=Path, required=True, help="JSONL, one line per CV")
    ap.add_argument("--ls-tasks", type=Path, help="also write Label Studio NER tasks (1 task per section)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs = (sorted(p for p in args.input.iterdir() if p.suffix.lower() in (".pdf", ".docx"))
            if args.input.is_dir() else [args.input])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    with open(args.out, "w", encoding="utf-8") as f:
        for pdf in docs:
            secs = parse_document(pdf, cfg)
            f.write(json.dumps({"cv": pdf.stem, "sections": secs}, ensure_ascii=False) + "\n")
            tasks += [{"data": {"cv": pdf.stem, "section": s["section_label"], "page": s["page"],
                                "text": s["text"]}} for s in secs if s["text"].strip()]
            print(f"{pdf.name}: " + ", ".join(s["section_label"] for s in secs))
    if args.ls_tasks:
        args.ls_tasks.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(tasks)} NER tasks -> {args.ls_tasks}")


def _selfcheck() -> None:
    cfg = load_config()
    kw, r = cfg["keywords"], cfg["heading"]["fuzzy_ratio"]
    assert fold("Học Vấn & Chứng chỉ") == "hoc van chung chi"
    assert match_label("HỌC VẤN & CHỨNG CHỈ", kw, r) == "EDUCATION"
    assert match_label("Kinh nghiệm làm việc", kw, r) == "EXPERIENCE"
    assert match_label("Experiance", kw, r) == "EXPERIENCE"  # misspelled
    assert match_label("Kỹ năng liên quan", kw, r) == "SKILLS"
    assert match_label("Cao đẳng Ngoại ngữ", kw, r) is None
    assert match_label("Dự án Thiết kế Thương hiệu Mới", kw, r) is None
    assert match_label("Nguyễn Văn A", kw, r) is None
    assert match_label("K I N H N G H I ỆM L À M V I ỆC", kw, r) == "EXPERIENCE"
    def ln(t: str, size: float = 10.0, bold: bool = False) -> PdfLine:
        return PdfLine(t, (0, 0, 0, 0), size, bold, 0)

    name = fold("Nguyễn Văn An")
    assert header_label(ln("Nguyễn Văn An", 24, True), "", None, name, cfg) == "NAME"
    assert header_label(ln("Kỹ sư phần mềm", 14), "NAME", ln("x", 24, True), name, cfg) == "JOB_TITLE"
    assert header_label(ln("Email: a@b.vn"), "JOB_TITLE", ln("x", 14), name, cfg) == "CONTACT"
    assert header_label(ln("Tôi là kỹ sư có năm năm kinh nghiệm"), "CONTACT", ln("x"), name, cfg) == "SUMMARY"


if __name__ == "__main__":
    _selfcheck() if len(sys.argv) == 1 else main()
