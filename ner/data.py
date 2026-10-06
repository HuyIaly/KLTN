"""Label Studio NER export -> word-level BIO examples -> model inputs.

Expected export (JSON, one task per CV section, as written by `cvparse.pdf_layout --ls-tasks`):
    [{"data": {"cv": "CV_1", "section": "EDUCATION", "text": "..."},
      "annotations": [{"result": [{"value": {"start": 0, "end": 12, "labels": ["SCHOOL"]}}]}]}]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class Example:
    cv: str
    section: str
    words: list[str]
    tags: list[str]


def bio_tags(text: str, spans: list[tuple[int, int, str]]) -> tuple[list[str], list[str]]:
    """Split on words/punctuation; a word inside a span gets B-/I- of that span."""
    words, tags, prev_span = [], [], None
    for m in TOKEN_RE.finditer(text):
        span = next((sp for sp in spans if sp[0] <= m.start() < sp[1]), None)
        tag = "O" if span is None else f"{'I' if span == prev_span else 'B'}-{span[2]}"
        words.append(m.group())
        tags.append(tag)
        prev_span = span
    return words, tags


def load_ls_export(path: Path) -> list[Example]:
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = []
    for task in tasks:
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue  # unlabeled task
        data = task["data"]
        spans = [(r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
                 for r in anns[-1]["result"] if r.get("type", "labels") == "labels"]
        words, tags = bio_tags(data["text"], spans)
        if words:
            examples.append(Example(str(data.get("cv", task.get("id"))), data.get("section", "OTHER"),
                                    words, tags))
    return examples


def split_by_cv(examples: list[Example], seed: int, dev: float = 0.1, test: float = 0.1):
    """All sections of one CV land in the same split (no leakage between train and test)."""
    cvs = sorted({e.cv for e in examples})
    random.Random(seed).shuffle(cvs)
    n_test, n_dev = round(len(cvs) * test), round(len(cvs) * dev)
    test_cv, dev_cv = set(cvs[:n_test]), set(cvs[n_test:n_test + n_dev])
    pick = lambda keep: [e for e in examples if keep(e.cv)]  # noqa: E731
    return (pick(lambda c: c not in test_cv and c not in dev_cv),
            pick(lambda c: c in dev_cv), pick(lambda c: c in test_cv))


def label_list(examples: list[Example]) -> list[str]:
    types = sorted({t[2:] for e in examples for t in e.tags if t != "O"})
    return ["O"] + [f"{p}-{t}" for t in types for p in "BI"]


class NerDataset(Dataset):
    """Each item is one chunk of <= max_len subwords. CRF runs on first subwords only.

    With use_section=True every chunk starts with the section name ("EDUCATION :"), a cheap
    stand-in for the section-title context that helped in Wei et al. 2020 (tagged O).
    """

    def __init__(self, examples: list[Example], tokenizer, label2id: dict[str, int],
                 max_len: int = 256, use_section: bool = True) -> None:
        self.items: list[dict] = []
        budget = max_len - 2  # [CLS] ... [SEP]
        for ex in examples:
            prefix = [ex.section, ":"] if use_section else []
            pieces = [tokenizer.tokenize(w) or [tokenizer.unk_token] for w in ex.words]
            pre_pieces = [tokenizer.tokenize(w) for w in prefix]
            start = 0
            while start < len(ex.words):
                used, end = sum(map(len, pre_pieces)), start
                while end < len(ex.words) and used + len(pieces[end]) <= budget:
                    used += len(pieces[end])
                    end += 1
                end = max(end, start + 1)  # a single over-long word still makes progress
                self.items.append(self._encode(tokenizer, pre_pieces + pieces[start:end],
                                               ["O"] * len(prefix) + ex.tags[start:end],
                                               label2id, max_len))
                start = end

    @staticmethod
    def _encode(tokenizer, word_pieces, tags, label2id, max_len) -> dict:
        ids, word_index = [tokenizer.cls_token_id], []
        for wp in word_pieces:
            word_index.append(len(ids))
            ids += tokenizer.convert_tokens_to_ids(wp)
        ids = ids[: max_len - 1] + [tokenizer.sep_token_id]
        word_index = [i for i in word_index if i < max_len - 1]
        return {"input_ids": ids, "word_index": word_index,
                "tags": [label2id.get(t, 0) for t in tags[: len(word_index)]]}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        t_len = max(len(b["input_ids"]) for b in batch)
        w_len = max(len(b["word_index"]) for b in batch)
        out = {k: torch.zeros(len(batch), n, dtype=torch.long)
               for k, n in (("input_ids", t_len), ("attention_mask", t_len),
                            ("word_index", w_len), ("tags", w_len))}
        out["input_ids"].fill_(pad_id)
        out["word_mask"] = torch.zeros(len(batch), w_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            n, w = len(b["input_ids"]), len(b["word_index"])
            out["input_ids"][i, :n] = torch.tensor(b["input_ids"])
            out["attention_mask"][i, :n] = 1
            out["word_index"][i, :w] = torch.tensor(b["word_index"])
            out["tags"][i, :w] = torch.tensor(b["tags"])
            out["word_mask"][i, :w] = True
        return out
    return collate
