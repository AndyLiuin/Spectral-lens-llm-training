#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_sa_lswa_1gpu.py

Spectrum analysis for checkpoints produced by 17_lswa.py:
- Activation covariance spectrum (per-layer, RMSNorm-normalized activations)
- Gradient SVD spectrum (per-layer parameter gradients)

Key differences from other spectrum analysis scripts:
  1) LSWA (Long-Short Window Attention): two masks per forward pass
     — long_bm: sliding_window_num_blocks
     — short_bm: sliding_window_num_blocks // 2
     Per-layer schedule: [long, short, short, short, ...]
     Decoder schedule = reversed encoder schedule.
  2) Adjusted window schedule (same as 16_adj):
     window_tokens_raw = 1728.0 * wfrac; sw_blocks = ceil(raw / block_size), clamped [1, 14]
  3) window_warmup_steps defaults to 2500 (not 3000)
  4) FP8 lm_head: uses FP8 compute path matching training for gradient fidelity
  5) lm_head_softcap defaults to 15.0

Includes:
- FP8 lm_head matching training (requires H100/H200 GPU)
- Wraparound-safe dataloader (stride = seq_len + 1)
- Strict checkpoint loading with prefix stripping
- args.json fallback for window schedule, softcap, dtype, model dims
- Autocast matching training dtype
"""

import os
import re
import gc
import json
import math
import pathlib
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import sys

HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import tqdm

# --- Flex Attention ---
try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    flex_attention = torch.compile(flex_attention, dynamic=False)
except Exception as e:
    flex_attention = None
    BlockMask = None
    print(f"[warn] flex_attention/BlockMask not available: {e}")

EOS_ID = 50256


# ==============================================================================
# 1a) FP8 LM head infrastructure (copied from training script)
# ==============================================================================

from torch import Tensor

def as_col_major(t: Tensor) -> Tensor:
    return t.t().contiguous().t()

if not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
    if not HELP_REQUESTED:
        raise RuntimeError(
            "This script requires torch.library.custom_op plus FP8-capable PyTorch support."
        )
    if not hasattr(torch, "library"):
        class _CompatLibrary:
            pass
        torch.library = _CompatLibrary()
    class _HelpOnlyCustomOp:
        def __call__(self, *args, **kwargs):
            def decorator(fn):
                def _unavailable(*_args, **_kwargs):
                    raise RuntimeError(
                        "This script requires torch.library.custom_op plus FP8-capable PyTorch support."
                    )
                _unavailable.__name__ = fn.__name__
                _unavailable.__doc__ = fn.__doc__
                def register_fake(fake_fn):
                    return fake_fn
                def register_autograd(*_args, **_kwargs):
                    return None
                _unavailable.register_fake = register_fake
                _unavailable.register_autograd = register_autograd
                return _unavailable
            return decorator
    torch.library.custom_op = _HelpOnlyCustomOp()

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(_x: Tensor, _w: Tensor):
        assert _x.is_contiguous() and _w.is_contiguous()
        x_f8 = _x.mul(x_s).to(torch.float8_e4m3fn)
        w_f8 = _w.mul(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8, w_f8.t(),
            out_dtype=torch.bfloat16,
            scale_a=_x.new_tensor(1 / x_s, dtype=torch.float32),
            scale_b=_x.new_tensor(1 / w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8
    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    torch._check(x.ndim == 2 and w.ndim == 2)
    torch._check(x.shape[1] == w.shape[1])
    torch._check(x.device == w.device)
    out = x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)
    x_f8 = x.to(torch.float8_e4m3fn)
    w_f8 = w.to(torch.float8_e4m3fn)
    return out, x_f8, w_f8

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor,
                   x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, _x_f8: Tensor, _w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(1 / x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(1 / w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(1 / grad_s, dtype=torch.float32)
        grad_f8 = grad.mul(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8.contiguous(), as_col_major(_w_f8),
            out_dtype=torch.bfloat16, scale_a=grad_inv_s, scale_b=w_inv_s, use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            grad_f8.t().contiguous(), as_col_major(_x_f8),
            out_dtype=torch.float32, scale_a=grad_inv_s, scale_b=x_inv_s, use_fast_accum=False,
        )
        return grad_x, grad_w
    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def _mm_backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(grad_out, x_f8, w_f8, x_s, w_s, grad_s)
    return grad_x, grad_w, None, None, None

def _mm_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(_mm_backward, setup_context=_mm_setup_context)

def lm_head_fp8(x: Tensor, w: Tensor) -> Tensor:
    _x = x.flatten(0, -2)
    out: Tensor = torch.ops.nanogpt.mm(_x, w, x_s=2.0, w_s=32.0, grad_s=2.0**29)[0]
    return out.reshape(*x.shape[:-1], -1)


# ==============================================================================
# 1b) Model (must match 17_lswa.py)
# ==============================================================================

def apply_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    """
    Truncated/"weird" RoPE:
      inv_freq = (1/1024) ** linspace(0,1, steps=dim//4), then padded with zeros.
    """
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0
        assert head_dim % 4 == 0

        inv_freq = (1.0 / 1024.0) ** torch.linspace(
            0.0, 1.0, steps=head_dim // 4, dtype=torch.float32
        )
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(head_dim // 4)])

        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i, j -> ij", t, inv_freq)

        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        cos = self.cos[None, :T, None, :]
        sin = self.sin[None, :T, None, :]
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], block_mask: BlockMask) -> torch.Tensor:
        B, T, _ = x.shape
        assert B == 1

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)

        if ve is None:
            v = self.mix[0] * v
        else:
            if ve.ndim == 2:
                ve = ve[None]
            ve = ve.view(B, T, self.num_heads, self.head_dim)
            v = self.mix[0] * v + self.mix[1] * ve.to(dtype=v.dtype)

        q, k = apply_norm(q), apply_norm(k)
        q = self.rotary(q)
        k = self.rotary(k)

        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, layer_idx: int, max_seq_len: int, skip_attn_layer: int = 7):
        super().__init__()
        self.layer_idx = layer_idx
        self.skip_attn_layer = skip_attn_layer
        self.attn = CausalSelfAttention(dim, n_head, max_seq_len) if layer_idx != skip_attn_layer else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], x0: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(apply_norm(x), ve, block_mask)
        x = x + self.mlp(apply_norm(x))
        return x


class ValueTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int):
        super().__init__()
        self.n_layer = n_layer
        self.emb = nn.ModuleList([nn.Embedding(vocab_size, n_embd) for _ in range(3)])

    def forward(self, tokens_1d: torch.Tensor) -> List[Optional[torch.Tensor]]:
        ve0 = self.emb[0](tokens_1d)
        ve1 = self.emb[1](tokens_1d)
        ve2 = self.emb[2](tokens_1d)
        ve = [ve0, ve1, ve2, None, None, None, None, None, None, ve0, ve1, ve2]
        assert len(ve) == self.n_layer, f"VE pattern length {len(ve)} != n_layer {self.n_layer}"
        return ve


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    seq_length: int = 65536
    block_size: int = 128


class GPT(nn.Module):
    """
    LSWA model. Uses lm_head_fp8() for the final projection, matching training.
    The weight matrix is stored as standard nn.Linear; FP8 quantization is applied
    at compute time (on-the-fly) as in training.
    """
    def __init__(self, config: GPTConfig, lm_head_softcap: float = 15.0):
        super().__init__()
        self.config = config
        self.lm_head_softcap = float(lm_head_softcap)
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([
                Block(config.n_embd, config.n_head, i, max_seq_len=config.seq_length, skip_attn_layer=7)
                for i in range(config.n_layer)
            ]),
        ))
        self.value_embeds = ValueTokenEmbedding(config.vocab_size, config.n_embd, config.n_layer)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    # -------------------------------------------------------------------
    # LSWA: two block masks (long / short)
    # -------------------------------------------------------------------
    def _make_long_short_block_masks(self, tokens_1d: torch.Tensor, sliding_window_num_blocks: int) -> Tuple[BlockMask, BlockMask]:
        BLOCK = self.config.block_size
        T = tokens_1d.numel()
        assert T % BLOCK == 0, f"T={T} must be divisible by block_size={BLOCK}"
        n_blocks = T // BLOCK

        docs = (tokens_1d == EOS_ID).cumsum(0)
        docs_low = docs.view(n_blocks, BLOCK)[:, 0].contiguous()
        docs_high = docs.view(n_blocks, BLOCK)[:, -1].contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

        block_idx = torch.arange(n_blocks, dtype=torch.int32, device=tokens_1d.device)
        q_idx = block_idx[:, None]
        kv_idx = block_idx[None, :]

        document_bm = (docs_low[:, None] <= docs_high[None, :]) & (docs_low[None, :] <= docs_high[:, None])
        document_full_bm = (docs_low[:, None] == docs_high[None, :]) & (docs_low[None, :] == docs_high[:, None])
        causal_bm = q_idx >= kv_idx
        causal_full_bm = q_idx > kv_idx

        nonzero_bm = causal_bm & document_bm
        full_bm = causal_full_bm & document_full_bm

        def dense_to_ordered(dense_mask):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)

        sw_long = torch.as_tensor(sliding_window_num_blocks, dtype=torch.int32, device=tokens_1d.device).clamp_min(1)
        sw_short = (sw_long // 2).clamp_min(1)

        def build_bm(sw_num_blocks):
            return BlockMask.from_kv_blocks(
                torch.clamp_max(kv_num_blocks, torch.clamp_min(sw_num_blocks - full_kv_num_blocks, 1)),
                kv_indices,
                torch.clamp_max(full_kv_num_blocks, sw_num_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK,
                mask_mod=mask_mod,
            )

        return build_bm(sw_long), build_bm(sw_short)

    def _layer_blockmask_schedule(self, long_bm: BlockMask, short_bm: BlockMask, n: int) -> List[BlockMask]:
        """
        [long, short, short, short, long, short, short, short, ...] truncated to n.
        """
        base = [long_bm, short_bm, short_bm, short_bm]
        out: List[BlockMask] = []
        while len(out) < n:
            out.extend(base)
        return out[:n]

    def forward(self, idx_1d: torch.Tensor, targets_1d: Optional[torch.Tensor], sliding_window_num_blocks: int, mask_block_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx_1d = idx_1d.view(-1)
        if targets_1d is not None:
            targets_1d = targets_1d.view(-1)

        long_bm, short_bm = self._make_long_short_block_masks(idx_1d, sliding_window_num_blocks)

        x = self.transformer.wte(idx_1d[None])
        x = apply_norm(x)
        x0 = x

        ve_all = self.value_embeds(idx_1d)

        # Encoder with LSWA schedule
        enc_masks = self._layer_blockmask_schedule(long_bm, short_bm, self.num_encoder_layers)
        skip_connections: List[torch.Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_all[i], x0, enc_masks[i])
            skip_connections.append(x)

        # Decoder: reverse schedule
        dec_masks = list(reversed(enc_masks))
        if self.num_decoder_layers != self.num_encoder_layers:
            dec_masks = self._layer_blockmask_schedule(long_bm, short_bm, self.num_decoder_layers)
            dec_masks.reverse()

        for i in range(self.num_decoder_layers):
            skip = skip_connections.pop()
            lid = self.num_encoder_layers + i
            x = self.transformer.h[lid](x + self.skip_weights[i] * skip, ve_all[lid], x0, dec_masks[i])

        x = apply_norm(x)
        # FP8 lm_head matching training
        logits = lm_head_fp8(x, self.lm_head.weight)
        sc = self.lm_head_softcap
        logits = sc * torch.tanh(logits / sc)
        logits = logits.float()

        loss = None
        if targets_1d is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_1d.view(-1))
        return logits, loss


# ==============================================================================
# 2) Layer runner (LSWA masks)
# ==============================================================================

@torch.no_grad()
def _embed_and_build_lswa_masks(model: GPT, tokens_1d: torch.Tensor, sliding_window_num_blocks: int):
    """Returns (x, x0, ve_all, enc_masks, dec_masks)."""
    long_bm, short_bm = model._make_long_short_block_masks(tokens_1d, sliding_window_num_blocks)

    x = model.transformer.wte(tokens_1d[None])
    x = apply_norm(x)
    x0 = x
    ve_all = model.value_embeds(tokens_1d)

    enc_masks = model._layer_blockmask_schedule(long_bm, short_bm, model.num_encoder_layers)
    dec_masks = list(reversed(enc_masks))
    if model.num_decoder_layers != model.num_encoder_layers:
        dec_masks = model._layer_blockmask_schedule(long_bm, short_bm, model.num_decoder_layers)
        dec_masks.reverse()

    return x, x0, ve_all, enc_masks, dec_masks


@torch.no_grad()
def run_until_layer_lswa(model: GPT, tokens_1d: torch.Tensor, layer_idx: int, sliding_window_num_blocks: int) -> torch.Tensor:
    x, x0, ve_all, enc_masks, dec_masks = _embed_and_build_lswa_masks(model, tokens_1d, sliding_window_num_blocks)

    skip_connections: List[torch.Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.transformer.h[i](x, ve_all[i], x0, enc_masks[i])
        skip_connections.append(x)
        if i == layer_idx:
            return apply_norm(x)

    for i in range(model.num_decoder_layers):
        skip = skip_connections.pop()
        lid = model.num_encoder_layers + i
        x = model.transformer.h[lid](x + model.skip_weights[i] * skip, ve_all[lid], x0, dec_masks[i])
        if lid == layer_idx:
            return apply_norm(x)

    raise RuntimeError(f"layer_idx {layer_idx} out of range")


# ==============================================================================
# 3) Online covariance + gradient SVD
# ==============================================================================

class OnlineCov:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, X: np.ndarray):
        if X.size == 0:
            return
        n_new = X.shape[0]
        mean_new = X.mean(axis=0)
        delta = mean_new - self.mean
        total = self.n + n_new

        self.M2 += (X.T @ X) - n_new * np.outer(mean_new, mean_new)
        if self.n > 0:
            self.M2 += (self.n * n_new / total) * np.outer(delta, delta)

        self.mean += (n_new / total) * delta
        self.n = total

    def covariance(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        return self.M2 / (self.n - 1)


@torch.no_grad()
def compute_covariance_spectrum(
    model: GPT,
    data_loader,
    layer_idx: int,
    sliding_window_num_blocks: int,
    autocast_ctx,
) -> np.ndarray:
    model.eval()
    cov = OnlineCov(model.config.n_embd)

    for x_batch_cpu, _y_batch_cpu in data_loader:
        B = x_batch_cpu.size(0)
        for b in range(B):
            tokens_1d = x_batch_cpu[b].to(next(model.parameters()).device, non_blocking=True).contiguous()
            with autocast_ctx:
                h = run_until_layer_lswa(model, tokens_1d, layer_idx, sliding_window_num_blocks)
            if not torch.isfinite(h).all():
                continue
            cov.update(h.reshape(-1, model.config.n_embd).float().cpu().numpy())

    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals


def _get_param_by_path(layer: nn.Module, path: str) -> torch.nn.Parameter:
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        if hasattr(obj, "requires_grad"):
            return obj
        raise TypeError(f"Resolved '{path}' is not a Parameter/Tensor")
    return obj


def compute_gradient_svd_spectrum(
    model: GPT,
    data_loader,
    layer_idx: int,
    param_path: str,
    num_samples: int,
    sliding_window_num_blocks: int,
    mask_block_size: int,
    autocast_ctx,
) -> np.ndarray:
    model.train()

    theta = _get_param_by_path(model.transformer.h[layer_idx], param_path)

    for p in model.parameters():
        p.requires_grad_(False)
    theta.requires_grad_(True)

    P = theta.numel()
    G = torch.zeros((num_samples, P), dtype=torch.float32, device="cpu")

    collected = 0
    for x_batch_cpu, y_batch_cpu in data_loader:
        if collected >= num_samples:
            break
        B = x_batch_cpu.size(0)
        for b in range(B):
            if collected >= num_samples:
                break
            tokens_1d = x_batch_cpu[b].to(next(model.parameters()).device, non_blocking=True).contiguous()
            targets_1d = y_batch_cpu[b].to(next(model.parameters()).device, non_blocking=True).contiguous()

            model.zero_grad(set_to_none=True)
            with autocast_ctx:
                _logits, loss = model(tokens_1d, targets_1d, sliding_window_num_blocks, mask_block_size)
            if loss is None or (not torch.isfinite(loss)):
                continue
            loss.backward()

            if theta.grad is None:
                continue
            grad_flat = theta.grad.detach().reshape(-1).to("cpu", non_blocking=True)
            G[collected, :] = grad_flat
            collected += 1

    if collected == 0:
        return np.array([], dtype=np.float64)

    with torch.no_grad():
        s = torch.linalg.svdvals(G[:collected].float())
    return s.cpu().numpy()


def analyze_power_law(spectrum: np.ndarray, tail_start: int, tail_finish: int) -> Dict[str, float]:
    spectrum = np.asarray(spectrum)
    spectrum = spectrum[np.isfinite(spectrum) & (spectrum > 0)]
    if spectrum.size == 0:
        return {"alpha": float("nan"), "r_squared": float("nan")}

    spectrum = np.sort(spectrum)[::-1]
    i = np.arange(1, len(spectrum) + 1)

    s_idx = max(0, min(len(spectrum) - 2, tail_start - 1))
    e_idx = min(len(spectrum), tail_finish) if tail_finish > 0 else len(spectrum)
    y = spectrum[s_idx:e_idx]
    x = i[s_idx:e_idx]

    if y.size > 2:
        logy = np.log(y)
        mask = np.isfinite(logy)
        if mask.sum() >= 2:
            slope, _, r_val, _, _ = linregress(np.log(x[mask]), logy[mask])
            return {"alpha": float(-slope), "r_squared": float(r_val ** 2)}

    return {"alpha": float("nan"), "r_squared": float("nan")}


# ==============================================================================
# 4) Wraparound-safe dataloader (streamshift)
# ==============================================================================

HEADER_BYTES = 256 * 4
FW_MAGIC = 20240520

def _load_data_shard(path: str) -> np.ndarray:
    try:
        if os.path.getsize(path) >= HEADER_BYTES:
            with open(path, "rb") as f:
                header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
            if header.size >= 3 and header[0] == FW_MAGIC:
                ntok = int(header[2])
                with open(path, "rb") as f:
                    f.seek(HEADER_BYTES)
                    data_bytes = os.path.getsize(path) - HEADER_BYTES
                    dtype = np.uint32 if data_bytes == ntok * 4 else np.uint16
                    tokens = np.frombuffer(f.read(), dtype=dtype)
                return np.asarray(tokens)
    except Exception:
        pass
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_data_loader_streamshift(
    data_path: str,
    num_samples: int,
    batch_size: int,
    seq_length: int,
):
    """
    Wraparound fix: read stride = (seq_length + 1) tokens, so y is the true next-token.
    """
    raw = _load_data_shard(data_path)
    stride = seq_length + 1
    num_sequences = len(raw) // stride
    if num_samples <= 0 or num_samples > num_sequences:
        num_samples = num_sequences

    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        start = i * stride
        end = start + bs * stride
        chunk = np.asarray(raw[start:end])
        chunk = chunk.reshape(bs, stride)

        t = torch.from_numpy(chunk).to(torch.int64)
        x = t[:, :-1].contiguous()
        y = t[:, 1:].contiguous()
        yield x, y


# ==============================================================================
# 5) Checkpoint loading + args.json fallback + ADJUSTED window schedule
# ==============================================================================

def _strip_prefix(k: str) -> str:
    for p in ("_orig_mod.", "module.", "model."):
        if k.startswith(p):
            return k[len(p):]
    return k


def load_training_args(ckpt_dir: pathlib.Path) -> Dict[str, Any]:
    candidates = [ckpt_dir / "args.json", ckpt_dir.parent / "args.json"]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def infer_batch_size_from_checkpoint_dir(checkpoint_dir: pathlib.Path) -> Optional[int]:
    for candidate in [checkpoint_dir, *checkpoint_dir.parents]:
        m = re.search(r"(?:^|_)bs(\d+)(?:_|$)", candidate.name)
        if m:
            return int(m.group(1))
    return None

def resolve_window_warmup_steps(
    checkpoint_dir: pathlib.Path,
    fallback_window_warmup_steps: int,
    reference_batch_size: int,
    train_args: Dict[str, Any],
) -> Tuple[int, str]:
    inferred_bs = infer_batch_size_from_checkpoint_dir(checkpoint_dir)
    if inferred_bs is not None and inferred_bs > 0 and reference_batch_size > 0:
        scaled = fallback_window_warmup_steps * reference_batch_size / inferred_bs
        warmup_steps = max(1, int(round(scaled)))
        source = f"checkpoint_dir batch size bs{inferred_bs} (reference bs{reference_batch_size} -> warmup {warmup_steps})"
        return warmup_steps, source

    warmup_steps = int(train_args.get("window_warmup_steps", fallback_window_warmup_steps))
    return warmup_steps, "args.json/CLI fallback"


def calculate_adjusted_window_blocks(step: int, window_warmup_steps: int, block_size: int) -> Tuple[int, int]:
    """
    Match training's adjusted window schedule (17_lswa.py / 16_fp8lmhead_win_adj.py):

        wfrac = min(step / warmup, 1.0)
        window_tokens_raw = 1728.0 * wfrac
        sw_blocks = ceil(window_tokens_raw / block_size)
        sw_blocks = clamp(sw_blocks, 1, 14)
        window_tokens = sw_blocks * block_size

    Returns (window_tokens, sw_blocks).
    """
    if window_warmup_steps <= 0:
        wfrac = 1.0
    else:
        wfrac = min(step / max(1, window_warmup_steps), 1.0)

    window_tokens_raw = 1728.0 * wfrac
    sw_blocks = int(math.ceil(window_tokens_raw / block_size))
    sw_blocks = max(1, min(sw_blocks, 14))

    window_tokens = sw_blocks * block_size
    return window_tokens, sw_blocks


def load_model_from_checkpoint(checkpoint_path: str, device: str, cfg: GPTConfig, lm_head_softcap: float = 15.0) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = GPT(cfg, lm_head_softcap=lm_head_softcap)

    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    if state_dict is None and isinstance(ckpt, dict):
        tensor_only = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensor_only:
            state_dict = tensor_only

    if state_dict is None:
        raise RuntimeError(f"No state_dict found in checkpoint: {checkpoint_path}")

    cleaned = {_strip_prefix(k): v for k, v in state_dict.items() if torch.is_tensor(v)}

    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict load reported missing/unexpected. missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        tmp = model.state_dict()
        ck = set(cleaned.keys())
        mk = set(tmp.keys())
        missing_keys = sorted(list(mk - ck))[:50]
        unexpected_keys = sorted(list(ck - mk))[:50]
        raise RuntimeError(
            f"Strict load FAILED for {checkpoint_path}\n"
            f"Error: {e}\n"
            f"Missing (first 50): {missing_keys}\n"
            f"Unexpected (first 50): {unexpected_keys}\n"
        )

    del ckpt, state_dict, cleaned
    gc.collect()

    model.to(device)
    model.eval()
    return model


def parse_layers(layers_str: str, total_layers: int) -> List[int]:
    if layers_str.lower() == "all":
        return list(range(total_layers))
    out = set()
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.update(range(a, b + 1))
        elif part:
            out.add(int(part))
    return sorted([l for l in out if 0 <= l < total_layers])


# ==============================================================================
# 6) Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spectrum Analysis (17_lswa checkpoints)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--validation_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="0,3,6,9,11")
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight")
    parser.add_argument("--task_index", type=int, default=-1)

    parser.add_argument("--seq_length", type=int, default=65536)
    parser.add_argument("--num_samples_cov", type=int, default=512)
    parser.add_argument("--num_samples_grad", type=int, default=512)
    parser.add_argument("--cov_batch_size", type=int, default=4)
    parser.add_argument("--grad_batch_size", type=int, default=4)
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)

    # Adjusted window schedule defaults: window_warmup_steps is the bs8 reference schedule unless overridden.
    parser.add_argument("--window_warmup_steps", type=int, default=2500)
    parser.add_argument("--window_warmup_reference_batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)

    # softcap value
    parser.add_argument("--lm_head_softcap", type=float, default=15.0)

    # optional override dtype
    parser.add_argument("--dtype", type=str, default="", choices=["", "float32", "bfloat16", "float16"])

    args = parser.parse_args()

    if flex_attention is None or BlockMask is None:
        raise RuntimeError("This script requires torch.nn.attention.flex_attention.")

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = load_training_args(ckpt_dir)

    # args.json still supplies block_size; warmup can be inferred from the bs* checkpoint folder.
    window_warm, window_warm_source = resolve_window_warmup_steps(
        ckpt_dir,
        args.window_warmup_steps,
        args.window_warmup_reference_batch_size,
        train_args,
    )
    block_size = int(train_args.get("block_size", args.block_size))

    # softcap from args.json fallback
    lm_head_softcap = float(train_args.get("lm_head_softcap", args.lm_head_softcap))

    if args.seq_length % block_size != 0:
        raise ValueError(f"--seq_length={args.seq_length} must be multiple of block_size={block_size}")

    # dtype / autocast
    dtype_name = args.dtype if args.dtype else str(train_args.get("dtype", "bfloat16"))
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # model dims from args.json fallback
    cfg = GPTConfig(
        vocab_size=int(train_args.get("vocab_size", 50304)),
        n_layer=int(train_args.get("n_layer", 12)),
        n_head=int(train_args.get("n_head", 6)),
        n_embd=int(train_args.get("n_embd", 768)),
        seq_length=int(args.seq_length),
        block_size=int(block_size),
    )

    print(f"[info] device={device} dtype={dtype_name} seq_length={args.seq_length}")
    print(
        f"[info] LSWA ADJUSTED window schedule: 0→1728 tokens over {window_warm} steps "
        f"(source: {window_warm_source}) | block_size={block_size}"
    )
    print(f"[info] LSWA layer schedule: [long, short, short, short, ...] | short = long // 2")
    print(f"[info] lm_head_softcap={lm_head_softcap}")
    print(f"[info] samples: cov={args.num_samples_cov} grad={args.num_samples_grad} | batch: cov={args.cov_batch_size} grad={args.grad_batch_size}")
    print(f"[info] model: L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd}")

    # checkpoints
    ckpt_paths = sorted(
        list(ckpt_dir.glob("checkpoint_step*.pt"))
        + list(ckpt_dir.glob("ckpt_step_*.pt"))
        + list(ckpt_dir.glob("checkpoint_target*.pt")),
        key=lambda p: int((re.search(r"step_?(\d+)", p.stem) or re.search(r"(\d+)", p.stem)).group(1))
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if args.task_index != -1:
        if not 0 <= args.task_index < len(ckpt_paths):
            raise IndexError(f"--task_index {args.task_index} out of bounds (n={len(ckpt_paths)})")
        ckpt_paths = [ckpt_paths[args.task_index]]
        print(f"[info] array mode: processing only index {args.task_index}: {ckpt_paths[0].name}")

    layers_to_analyze = parse_layers(args.layers, cfg.n_layer)
    matrices_to_analyze = [m.strip() for m in args.matrices.split(",")]
    for path in tqdm(ckpt_paths, desc="Checkpoints"):
        step = int((re.search(r"step_?(\d+)", path.stem) or re.search(r"(\d+)", path.stem)).group(1))

        # ADJUSTED window schedule: linear 0 → 1728, ceil, clamp [1,14]
        window_tokens, window_blocks = calculate_adjusted_window_blocks(step, window_warm, block_size)

        print(f"\n[ckpt] step={step} window_tokens={window_tokens} window_blocks={window_blocks} (long={window_blocks}, short={max(1, window_blocks // 2)})")

        # skip if everything exists (check OUTSIDE layer loop)
        all_exist = True
        for l in layers_to_analyze:
            cov_p = out_dir / f"cov_spectrum_{step}_L{l:02d}.npy"
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                grad_p = out_dir / f"grad_spectrum_{step}_L{l:02d}_{matrix_name_clean}.npy"
                weight_path = grad_p.with_name(grad_p.name.replace("grad_spectrum_", "weight_spectrum_"))
                if not (cov_p.exists() and grad_p.exists() and weight_path.exists()):
                    all_exist = False
                    break
            if not all_exist:
                break

        if all_exist:
            print("[info] all outputs exist; skipping.")
            continue

        # Load model (OUTSIDE layer loop)
        model = load_model_from_checkpoint(str(path), device=device, cfg=cfg, lm_head_softcap=lm_head_softcap)

        results = []
        for l in layers_to_analyze:
            cov_p = out_dir / f"cov_spectrum_{step}_L{l:02d}.npy"

            if cov_p.exists():
                cov_eigs = np.load(cov_p)
            else:
                dl_cov = get_data_loader_streamshift(
                    args.validation_data_path,
                    num_samples=args.num_samples_cov,
                    batch_size=args.cov_batch_size,
                    seq_length=args.seq_length,
                )
                cov_eigs = compute_covariance_spectrum(
                    model, dl_cov, layer_idx=l,
                    sliding_window_num_blocks=window_blocks,
                    autocast_ctx=autocast_ctx,
                )
                np.save(cov_p, cov_eigs)

            cov_fit = analyze_power_law(cov_eigs, args.tail_start, args.tail_finish)

            # Loop over matrices for gradient
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                grad_p = out_dir / f"grad_spectrum_{step}_L{l:02d}_{matrix_name_clean}.npy"
                weight_path = grad_p.with_name(grad_p.name.replace("grad_spectrum_", "weight_spectrum_"))

                if grad_p.exists():
                    grad_s = np.load(grad_p)
                else:
                    dl_grad = get_data_loader_streamshift(
                        args.validation_data_path,
                        num_samples=args.num_samples_grad,
                        batch_size=args.grad_batch_size,
                        seq_length=args.seq_length,
                    )
                    grad_s = compute_gradient_svd_spectrum(
                        model, dl_grad, layer_idx=l,
                        param_path=matrix_name,
                        num_samples=args.num_samples_grad,
                        sliding_window_num_blocks=window_blocks,
                        mask_block_size=block_size,
                        autocast_ctx=autocast_ctx,
                    )
                    np.save(grad_p, grad_s)

                grad_fit = analyze_power_law(grad_s, args.tail_start, args.tail_finish)

                weight_s = None
                if weight_path.exists():
                    weight_s = np.load(weight_path)
                else:
                    try:
                        weight_theta = _get_param_by_path(model.transformer.h[l], matrix_name)
                        weight_s = torch.linalg.svdvals(weight_theta.detach().float()).cpu().numpy()
                        np.save(weight_path, weight_s)
                    except Exception as e:
                        weight_s = None
                if weight_s is None:
                    weight_fit = {'alpha': float('nan'), 'r_squared': float('nan')}
                else:
                    weight_fit = analyze_power_law(weight_s, args.tail_start, args.tail_finish)

                results.append({
                    "step": step,
                    "layer": l,
                    "matrix": matrix_name,
                    "cov_alpha": float(cov_fit["alpha"]),
                    "cov_r2": float(cov_fit["r_squared"]),
                    "grad_alpha": float(grad_fit["alpha"]),
                    "grad_r2": float(grad_fit["r_squared"]),
                    "window_tokens": window_tokens,
                    "window_blocks": window_blocks,
                    'weight_alpha': float(weight_fit['alpha']),
                    'weight_r2': float(weight_fit['r_squared']),
                })

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if results:
            csv_path = out_dir / f"summary_step_{step}.csv"
            new_df = pd.DataFrame(results)
            if csv_path.exists():
                try:
                    existing_df = pd.read_csv(csv_path)
                    keys = ['step', 'layer', 'matrix']
                    combined_df = pd.concat([existing_df, new_df])
                    final_df = combined_df.drop_duplicates(subset=keys, keep='last').sort_values(by=keys)
                    final_df.to_csv(csv_path, index=False)
                except Exception as e:
                    fallback_path = csv_path.with_name(csv_path.stem + "_new.csv")
                    new_df.to_csv(fallback_path, index=False)
            else:
                new_df.to_csv(csv_path, index=False)
            print(f"[info] saved {csv_path.name}")

    print("[done] analysis complete.")


if __name__ == "__main__":
    main()
