from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SquaredMLP, apply_norm


class BaselineSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, c = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(bsz, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, t, c)
        return self.c_proj(y)


class BaselineBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn = BaselineSelfAttention(d_model=d_model, n_heads=n_heads)
        self.mlp = SquaredMLP(d_model=d_model, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(apply_norm(x))
        x = x + self.mlp(apply_norm(x))
        return x


class BaselineTokenLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)

        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, d_model))
        self.blocks = nn.ModuleList(
            [BaselineBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult) for _ in range(int(n_layers))]
        )
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x_tokens: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_embed(x_tokens)
        x = x + self.pos_embed[: x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        h = apply_norm(x)
        logits = self.lm_head(h)
        if return_repr:
            return logits, h
        return logits, None
