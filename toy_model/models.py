from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .variant_utils import parse_variant_tokens
except ImportError:
    from variant_utils import parse_variant_tokens


def rff_features_torch(z: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    # z: [B, T, d], omega: [P, d], phase: [P]
    bsz, seq_len, d = z.shape
    p = omega.shape[0]
    flat = z.reshape(bsz * seq_len, d)
    feats = torch.cos(flat @ omega.t() + phase.unsqueeze(0))
    feats = feats * (2.0 / p) ** 0.5
    return feats.reshape(bsz, seq_len, p)


def _rms_norm_lastdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RotaryCache(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head dimension.")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, H, Dh]
        t = x.shape[1]
        key = (x.device, x.dtype, t)
        if key != self._cache_key:
            self._cache_key = key
            inv_freq = self.inv_freq.to(device=x.device)
            pos = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(pos, inv_freq)
            cos = freqs.cos().to(dtype=x.dtype)
            sin = freqs.sin().to(dtype=x.dtype)
            self.cos_cached = cos[None, :, None, :]
            self.sin_cached = sin[None, :, None, :]
        return self.cos_cached, self.sin_cached


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, Dh], cos/sin: [1, T, 1, Dh/2]
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class ToySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_rope: bool = False,
        qk_rmsnorm: bool = False,
        window_size: int = 0,
        attention_scale: Optional[float] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qk_rmsnorm = qk_rmsnorm
        self.window_size = max(0, int(window_size))
        self.attention_scale = attention_scale

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryCache(self.head_dim) if self.use_rope else None

    def _attention_mask(self, t: int, device: torch.device) -> torch.Tensor:
        q = torch.arange(t, device=device).view(t, 1)
        k = torch.arange(t, device=device).view(1, t)
        causal = k <= q
        if self.window_size > 0:
            local = (q - k) < self.window_size
            mask = causal & local
        else:
            mask = causal
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = x.shape

        q = self.q_proj(x).view(bsz, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, t, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, t, self.n_heads, self.head_dim)

        if self.qk_rmsnorm:
            q = _rms_norm_lastdim(q)
            k = _rms_norm_lastdim(k)

        if self.use_rope:
            cos, sin = self.rope(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)  # [B, H, T, Dh]
        v = v.transpose(1, 2)  # [B, H, T, Dh]

        scale = self.attention_scale if self.attention_scale is not None else self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = self._attention_mask(t=t, device=x.device)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)  # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.out_proj(y)


class CausalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 2,
        use_rope: bool = False,
        qk_rmsnorm: bool = False,
        window_size: int = 0,
        attention_scale: Optional[float] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ToySelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_rope=use_rope,
            qk_rmsnorm=qk_rmsnorm,
            window_size=window_size,
            attention_scale=attention_scale,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerRFFRegressor(nn.Module):
    def __init__(
        self,
        d: int,
        p: int,
        seq_len: int,
        omega: torch.Tensor,
        phase: torch.Tensor,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        ff_mult: int = 2,
        variant: str = "baseline",
        window_size: int = 0,
        attention_scale: Optional[float] = None,
    ):
        super().__init__()
        self.d = d
        self.p = p
        self.seq_len = seq_len
        self.variant = variant

        self.register_buffer("omega", omega.clone().detach())
        self.register_buffer("phase", phase.clone().detach())

        variant_tokens = parse_variant_tokens(variant)
        use_rope = "rope" in variant_tokens
        qk_rmsnorm = "rope" in variant_tokens
        use_unet = "unet" in variant_tokens
        local_window = window_size if "fixed_window" in variant_tokens else 0
        attn_scale = attention_scale if "attn_scale" in variant_tokens else None

        self.use_pos_embed = not use_rope
        self.input_proj = nn.Linear(p, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model)) if self.use_pos_embed else None

        if use_unet and n_layers < 2:
            n_layers = 2
        self.n_layers = n_layers
        self.use_unet = use_unet

        self.blocks = nn.ModuleList(
            [
                CausalBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    use_rope=use_rope,
                    qk_rmsnorm=qk_rmsnorm,
                    window_size=local_window,
                    attention_scale=attn_scale,
                )
                for _ in range(n_layers)
            ]
        )

        if self.use_unet:
            dec_layers = n_layers // 2
            self.skip_weights = nn.Parameter(torch.ones(dec_layers))
        else:
            self.skip_weights = None

        self.readout = nn.Linear(d_model, 1)

        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_unet:
            for block in self.blocks:
                x = block(x)
            return x

        enc = self.n_layers // 2
        dec = self.n_layers - enc
        skips = []

        for i in range(enc):
            x = self.blocks[i](x)
            skips.append(x)

        for i in range(dec):
            j = enc + i
            x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[j](x)

        return x

    def forward(self, z: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        phi = rff_features_torch(z, self.omega, self.phase)
        x = self.input_proj(phi)
        if self.pos_embed is not None:
            x = x + self.pos_embed[: x.shape[1], :]

        x = self._forward_blocks(x)
        h = x[:, -1, :]
        pred = self.readout(h).squeeze(-1)
        if return_repr:
            return pred, h
        return pred, None


class LinearRFFRegressor(nn.Module):
    def __init__(self, d: int, p: int, omega: torch.Tensor, phase: torch.Tensor):
        super().__init__()
        self.d = d
        self.p = p
        self.register_buffer("omega", omega.clone().detach())
        self.register_buffer("phase", phase.clone().detach())
        self.readout = nn.Linear(p, 1)

    def forward(self, z: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        phi = rff_features_torch(z, self.omega, self.phase)
        h = phi.mean(dim=1)
        pred = self.readout(h).squeeze(-1)
        if return_repr:
            return pred, h
        return pred, None


def build_model(
    track: str,
    d: int,
    p: int,
    seq_len: int,
    omega: torch.Tensor,
    phase: torch.Tensor,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    ff_mult: int = 2,
    variant: str = "baseline",
    window_size: int = 0,
    attention_scale: Optional[float] = None,
) -> nn.Module:
    t = track.strip().lower()
    if t in {"a", "transformer", "transformer_rff"}:
        return TransformerRFFRegressor(
            d=d,
            p=p,
            seq_len=seq_len,
            omega=omega,
            phase=phase,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            variant=variant,
            window_size=window_size,
            attention_scale=attention_scale,
        )
    if t in {"b", "linear", "linear_rff"}:
        return LinearRFFRegressor(d=d, p=p, omega=omega, phase=phase)
    raise ValueError(f"Unknown track: {track}")
