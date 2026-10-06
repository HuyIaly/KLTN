"""Transformer encoder + CRF over first subwords (La et al. 2026, PAP_NER: CRF +0.4 F1, +0.96 on long entities)."""
from __future__ import annotations

import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel


class TransformerCRF(nn.Module):
    def __init__(self, encoder_name: str, num_tags: int, dropout: float = 0.1, use_crf: bool = True) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(self.encoder.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True) if use_crf else None  # None = softmax ablation

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  word_index: torch.Tensor) -> torch.Tensor:
        # input_ids/attention_mask: (B, T); word_index: (B, W) position of each word's first subword
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, H)
        h = h.gather(1, word_index.unsqueeze(-1).expand(-1, -1, h.size(-1)))                  # (B, W, H)
        return self.emission(self.dropout(h))                                                   # (B, W, num_tags)

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        em = self.emissions(batch["input_ids"], batch["attention_mask"], batch["word_index"])
        if self.crf is None:
            return nn.functional.cross_entropy(em[batch["word_mask"]], batch["tags"][batch["word_mask"]])
        # CRF in fp32: its log-sum-exp is unstable under autocast fp16
        return -self.crf(em.float(), batch["tags"], mask=batch["word_mask"], reduction="mean")

    @torch.no_grad()
    def decode(self, batch: dict[str, torch.Tensor]) -> list[list[int]]:
        em = self.emissions(batch["input_ids"], batch["attention_mask"], batch["word_index"]).float()
        if self.crf is None:
            return [row[m].tolist() for row, m in zip(em.argmax(-1), batch["word_mask"], strict=True)]
        return self.crf.decode(em, mask=batch["word_mask"])
