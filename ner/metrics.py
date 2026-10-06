"""Strict entity-level P/R/F1 (exact span + type), same as seqeval's default strict mode."""
from __future__ import annotations

from collections import Counter


def spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """BIO -> {(start, end_exclusive, type)}; a stray I- starts a new entity."""
    out, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if start is not None and (t == "O" or t.startswith("B-") or t[2:] != typ):
            out.add((start, i, typ))
            start = None
        if t != "O" and start is None:
            start, typ = i, t[2:]
    return out


def prf(gold: list[list[str]], pred: list[list[str]]) -> dict[str, dict[str, float]]:
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred, strict=True):
        gs, ps = spans(g), spans(p)
        for s in gs:
            n_gold[s[2]] += 1
        for s in ps:
            n_pred[s[2]] += 1
        for s in gs & ps:
            tp[s[2]] += 1

    def score(t: int, g: int, p: int) -> dict[str, float]:
        pr, rc = t / p if p else 0.0, t / g if g else 0.0
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": g}

    out = {typ: score(tp[typ], n_gold[typ], n_pred[typ]) for typ in sorted(n_gold | n_pred)}
    out["micro"] = score(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values()))
    return out


if __name__ == "__main__":
    assert spans(["B-ORG", "I-ORG", "O", "B-PER", "I-ORG"]) == {(0, 2, "ORG"), (3, 4, "PER"), (4, 5, "ORG")}
    r = prf([["B-ORG", "I-ORG", "O"]], [["B-ORG", "O", "O"]])
    assert r["micro"]["f1"] == 0.0 and r["ORG"]["support"] == 1
    assert prf([["B-A", "O"]], [["B-A", "O"]])["micro"]["f1"] == 1.0
    print("metrics ok")
