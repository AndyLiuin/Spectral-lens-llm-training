from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import apply_norm
from .rope import RopeBlock


class UntiedTokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.vocab_size = int(vocab_size)

        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [RopeBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult) for _ in range(int(n_layers))]
        )
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, x_tokens: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_embed(x_tokens)
        x = apply_norm(x)
        for block in self.blocks:
            x = block(x)
        h = apply_norm(x)
        logits = self.lm_head(h)
        if return_repr:
            return logits, h
        return logits, None
