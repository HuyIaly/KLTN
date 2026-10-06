"""Train / evaluate Transformer(-CRF) NER on a Label Studio export.

    python -m ner.train --export labelstudio/ner_export.json --model vinai/phobert-base-v2 --out runs/ner/phobert_crf
    python -m ner.train --export labelstudio/ner_export.json --model xlm-roberta-base     --out runs/ner/xlmr_crf
    python -m ner.train ... --no-crf        # softmax ablation
    python -m ner.train ... --no-section    # without the section prefix

Writes to --out: best.pt (state_dict + config), log.jsonl (one line per epoch), test_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import NerDataset, label_list, load_ls_export, make_collate, split_by_cv
from .metrics import prf
from .model import TransformerCRF

log = logging.getLogger("ner")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: TransformerCRF, loader: DataLoader, id2label: list[str], device: torch.device) -> dict:
    model.eval()
    gold, pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        for p, t, m in zip(model.decode(batch), batch["tags"], batch["word_mask"], strict=True):
            gold.append([id2label[i] for i in t[m].tolist()])
            pred.append([id2label[i] for i in p])
    return prf(gold, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help="Label Studio NER export (JSON)")
    ap.add_argument("--model", default="vinai/phobert-base-v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder LR (PAP_NER)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="emission + CRF LR (PAP_NER)")
    ap.add_argument("--max-len", type=int, default=256, help="PhoBERT max is 256")
    ap.add_argument("--patience", type=int, default=5, help="stop after N epochs without dev F1 gain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-crf", action="store_true")
    ap.add_argument("--no-section", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "train.log", "w", "utf-8")])
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = load_ls_export(args.export)
    train_ex, dev_ex, test_ex = split_by_cv(examples, args.seed)
    labels = label_list(examples)
    label2id = {t: i for i, t in enumerate(labels)}
    log.info("sections train/dev/test = %d/%d/%d, tags = %s", len(train_ex), len(dev_ex), len(test_ex), labels)

    # ponytail: syllable-level input; PhoBERT was pretrained on word-segmented text (RDRSegmenter),
    # add segmentation here if PhoBERT lags XLM-R on the dev set.
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = {name: NerDataset(ex, tok, label2id, args.max_len, use_section=not args.no_section)
          for name, ex in (("train", train_ex), ("dev", dev_ex), ("test", test_ex))}
    collate = make_collate(tok.pad_token_id)
    loaders = {name: DataLoader(d, batch_size=args.batch, shuffle=name == "train", collate_fn=collate,
                                pin_memory=device.type == "cuda")
               for name, d in ds.items()}

    model = TransformerCRF(args.model, len(labels), use_crf=not args.no_crf).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                                   {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * len(loaders["train"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "labels": labels}
    best_f1, bad_epochs = -1.0, 0
    with open(args.out / "log.jsonl", "w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for batch in loaders["train"]:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = model.loss(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total += loss.item()

            dev = evaluate(model, loaders["dev"], labels, device)["micro"]
            row = {"epoch": epoch, "train_loss": total / len(loaders["train"]), "dev": dev}
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            log.info("epoch %d loss %.4f dev F1 %.4f", epoch, row["train_loss"], dev["f1"])

            if dev["f1"] > best_f1:
                best_f1, bad_epochs = dev["f1"], 0
                torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
                           args.out / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    log.info("early stop at epoch %d", epoch)
                    break

    ckpt = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(model, loaders["test"], labels, device)
    (args.out / "test_metrics.json").write_text(json.dumps({"best_epoch": ckpt["epoch"], **test}, indent=2))
    log.info("best epoch %d, test micro F1 %.4f", ckpt["epoch"], test["micro"]["f1"])
    for typ, s in test.items():
        log.info("  %-15s P %.3f R %.3f F1 %.3f (n=%d)", typ, s["precision"], s["recall"], s["f1"], s["support"])


if __name__ == "__main__":
    main()
