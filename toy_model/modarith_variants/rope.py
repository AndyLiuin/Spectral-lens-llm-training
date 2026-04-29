from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import RotaryCache, SquaredMLP, apply_norm, apply_rotary_emb


class RopeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.c_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = RotaryCache(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = x.shape
        q = self.c_q(x).view(bsz, t, self.n_heads, self.head_dim)
        k = self.c_k(x).view(bsz, t, self.n_heads, self.head_dim)
        v = self.c_v(x).view(bsz, t, self.n_heads, self.head_dim)

        cos, sin = self.rotary(q)
        q = apply_norm(q)
        k = apply_norm(k)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)


class RopeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn = RopeSelfAttention(d_model=d_model, n_heads=n_heads)
        self.mlp = SquaredMLP(d_model=d_model, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(apply_norm(x))
        x = x + self.mlp(apply_norm(x))
        return x


class RopeTokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.vocab_size = int(vocab_size)

        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [RopeBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult) for _ in range(int(n_layers))]
        )
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, x_tokens: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_embed(x_tokens)
        for block in self.blocks:
            x = block(x)
        h = apply_norm(x)
        logits = self.lm_head(h)
        if return_repr:
            return logits, h
        return logits, None
