from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_norm(x: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),))
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)


class RotaryCache(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        key = (x.device, x.dtype, seq_len)
        if key != self._cache_key:
            self._cache_key = key
            inv_freq = self.inv_freq.to(device=x.device)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos().to(dtype=x.dtype)
            sin = freqs.sin().to(dtype=x.dtype)
            self.cos_cached = cos[None, :, None, :]
            self.sin_cached = sin[None, :, None, :]
        return self.cos_cached, self.sin_cached


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)


def causal_mask(t: int, device: torch.device, window_size: int = 0) -> torch.Tensor:
    q = torch.arange(t, device=device).view(t, 1)
    k = torch.arange(t, device=device).view(1, t)
    mask = k <= q
    if window_size > 0:
        mask = mask & ((q - k) < int(window_size))
    return mask


class SquaredMLP(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4):
        super().__init__()
        hidden_dim = int(ff_mult) * int(d_model)
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class ScaleStyleSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        use_rope: bool,
        use_value_mix: bool,
        window_size: int = 0,
        attention_scale: Optional[float] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.use_value_mix = bool(use_value_mix)
        self.window_size = max(0, int(window_size))
        self.attention_scale = attention_scale

        self.c_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_()
        self.lamb = nn.Parameter(torch.tensor(0.5, dtype=torch.float32)) if self.use_value_mix else None
        self.rotary = RotaryCache(self.head_dim) if use_rope else None

    def forward(self, x: torch.Tensor, v1: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, t, _ = x.shape

        q = self.c_q(x).view(bsz, t, self.n_heads, self.head_dim)
        k = self.c_k(x).view(bsz, t, self.n_heads, self.head_dim)
        v = self.c_v(x).view(bsz, t, self.n_heads, self.head_dim)

        if self.use_value_mix:
            if v1 is None:
                v1 = v
            v = (1.0 - self.lamb) * v + self.lamb * v1.view_as(v)

        q = apply_norm(q)
        k = apply_norm(k)
        if self.rotary is not None:
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.attention_scale if self.attention_scale is not None else self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = causal_mask(t=t, device=x.device, window_size=self.window_size)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.c_proj(y), v1


class ScaleStyleBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        *,
        use_rope: bool,
        use_value_mix: bool,
        use_x0_mix: bool,
        window_size: int = 0,
        attention_scale: Optional[float] = None,
    ):
        super().__init__()
        self.use_x0_mix = bool(use_x0_mix)
        self.attn = ScaleStyleSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_rope=use_rope,
            use_value_mix=use_value_mix,
            window_size=window_size,
            attention_scale=attention_scale,
        )
        self.mlp = SquaredMLP(d_model=d_model, ff_mult=ff_mult)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32)) if self.use_x0_mix else None

    def forward(self, x: torch.Tensor, v1: Optional[torch.Tensor], x0: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_x0_mix:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(apply_norm(x), v1)
        x = x + x1
        x = x + self.mlp(apply_norm(x))
        return x, v1
