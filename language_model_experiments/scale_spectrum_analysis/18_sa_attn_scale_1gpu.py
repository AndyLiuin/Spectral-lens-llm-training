#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_sa_attn_scale_1gpu.py

Spectrum analysis for checkpoints produced by 18_attn_scale.py:
- Activation covariance spectrum (per-layer, RMSNorm-normalized activations)
- Gradient SVD spectrum (per-layer parameter gradients)

Key differences from other spectrum analysis scripts:
  1) Uses `attn_scale = 0.12` in attention (instead of 1/sqrt(head_dim))
  2) Adjusted window schedule (same as 16_adj/17_lswa):
     window_tokens_raw = 1728.0 * wfrac; sw_blocks = ceil(raw / block_size), clamped [1, 14]
  3) FP8 lm_head: uses FP8 compute path matching training for gradient fidelity
  4) lm_head_softcap defaults to 15.0 (was 30.0 in old script)

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
# 1b) Model (must match 18_attn_scale.py)
# ==============================================================================

def apply_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    """
    Truncated/"weird" RoPE as in the reference script:
      inv_freq = (1/1024) ** linspace(0,1, steps=dim//4), then padded with zeros.
    """
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        assert head_dim % 4 == 0, f"reference truncated RoPE expects head_dim divisible by 4, got {head_dim}"

        inv_freq = (1.0 / 1024.0) ** torch.linspace(
            0.0, 1.0, steps=head_dim // 4, dtype=torch.float32
        )
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(head_dim // 4)])  # -> head_dim//2

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
        
        # KEY DIFFERENCE: attn_scale matches script 18 training
        self.attn_scale = 0.12

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

        # Uses self.attn_scale (0.12)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale)
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
    seq_length: int = 65536  # analysis seq length


class GPT(nn.Module):
    """
    Uses lm_head_fp8() for the final projection, matching training.
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

    def forward(self, idx_1d: torch.Tensor, targets_1d: Optional[torch.Tensor], sliding_window_num_blocks: int, mask_block_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, x0, ve_all, block_mask = embed_inputs_and_mask_sparse(
            self, idx_1d, sliding_window_num_blocks=sliding_window_num_blocks, mask_block_size=mask_block_size
        )

        skip_connections: List[torch.Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_all[i], x0, block_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            skip = skip_connections.pop()
            lid = self.num_encoder_layers + i
            # Match training: x = x + skip_weights[i] * skip, then pass through block
            x = self.transformer.h[lid](x + self.skip_weights[i] * skip, ve_all[lid], x0, block_mask)

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
# 2) Mask + layer runner (match training)
# ==============================================================================

def embed_inputs_and_mask_sparse(model: GPT, idx_1d: torch.Tensor, sliding_window_num_blocks: int, mask_block_size: int = 128):
    """
    Creates embeddings and the FlexAttention block mask.
    matches training logic:
      - sliding_window_num_blocks controls the causal window
      - we apply causal_bm & document_bm
    """
    # Ensure 1D — dataloader may return [1, T] or [T]
    if idx_1d.ndim == 2:
        assert idx_1d.size(0) == 1, f"Expected batch size 1, got {idx_1d.size(0)}"
        idx_1d = idx_1d[0]  # [T]
    assert idx_1d.ndim == 1, f"Expected 1D token sequence, got shape {tuple(idx_1d.shape)}"

    T = idx_1d.numel()

    # 1. Embeddings — match training: wte(inputs[None]) -> [1, T, C]
    x = model.transformer.wte(idx_1d[None])  # [1, T, C]
    x = apply_norm(x)
    x0 = x
    ve_all = model.value_embeds(idx_1d)  # list of [T, C] or None

    # 2. Block Mask — matches training _make_long_short_block_masks
    BLOCK = mask_block_size
    assert T % BLOCK == 0, f"T={T} must be divisible by BLOCK_SIZE={BLOCK}"
    n_blocks = T // BLOCK

    # doc ids per token
    docs = (idx_1d == EOS_ID).cumsum(0)
    docs_low  = docs.view(n_blocks, BLOCK)[:, 0].contiguous()
    docs_high = docs.view(n_blocks, BLOCK)[:, -1].contiguous()

    # token-level causal + same-doc constraint
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

    # block-level masks
    block_idx = torch.arange(n_blocks, dtype=torch.int32, device=idx_1d.device)
    q_blk = block_idx[:, None]
    kv_blk = block_idx[None, :]

    document_bm      = (docs_low[:, None] <= docs_high[None, :]) & (docs_low[None, :] <= docs_high[:, None])
    document_full_bm = (docs_low[:, None] == docs_high[None, :]) & (docs_low[None, :] == docs_high[:, None])
    causal_bm        = q_blk >= kv_blk
    causal_full_bm   = q_blk > kv_blk

    nonzero_bm = causal_bm & document_bm
    full_bm    = causal_full_bm & document_full_bm

    def dense_to_ordered(dense_mask: torch.Tensor):
        num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
        indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    kv_num_blocks,      kv_indices      = dense_to_ordered(nonzero_bm & ~full_bm)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)

    sw = torch.as_tensor(sliding_window_num_blocks, dtype=torch.int32, device=idx_1d.device).clamp_min(1)

    block_mask = BlockMask.from_kv_blocks(
        torch.clamp_max(kv_num_blocks,      torch.clamp_min(sw - full_kv_num_blocks, 1)),
        kv_indices,
        torch.clamp_max(full_kv_num_blocks, sw - 1),
        full_kv_indices,
        BLOCK_SIZE=BLOCK,
        mask_mod=mask_mod,
    )
    
    return x, x0, ve_all, block_mask


@torch.no_grad()
def run_until_layer(model: GPT, x: torch.Tensor, x0: torch.Tensor, ve_all: List[Optional[torch.Tensor]], block_mask: BlockMask, layer_idx: int) -> torch.Tensor:
    """
    Runs forward pass up to (and including) layer_idx, returning the normalized output of that layer.
    """
    skip_connections: List[torch.Tensor] = []

    # Encoder
    for i in range(model.num_encoder_layers):
        if i > layer_idx:
            # We haven't reached decoder yet, but stopped early in encoder
            # If we just finished layer_idx, x is the output of that layer.
            # But wait: the loop runs layer i, updates x.
            # So if layer_idx < i, we should have returned already.
            # Actually, let's just check at start of loop:
            # if we are about to run layer i, but layer_idx < i, we are done?
            # No, if layer_idx is 0, we run layer 0, then return output of layer 0.
            pass
        
        # Determine if we should run this layer
        # Global layers: 0..5 (encoder), 6..11 (decoder)
        if i > layer_idx:
            return apply_norm(x)

        x_out = model.transformer.h[i](x, ve_all[i], x0, block_mask)
        skip_connections.append(x_out)
        x = x_out
        
        if i == layer_idx:
            return apply_norm(x)

    # Decoder
    for i in range(model.num_decoder_layers):
        lid = model.num_encoder_layers + i
        if lid > layer_idx:
            return apply_norm(x)

        skip = skip_connections.pop()
        # x = x + skip_weights * skip
        # then block
        x = model.transformer.h[lid](x + model.skip_weights[i] * skip, ve_all[lid], x0, block_mask)
        
        if lid == layer_idx:
            return apply_norm(x)

    return apply_norm(x)


# ==============================================================================
# 3) Dataloader (Wraparound-safe, stride = seq_len + 1)
# ==============================================================================

def _load_data_shard(path: str):
    HEADER_BYTES = 256 * 4
    FW_MAGIC = 20240520
    try:
        if os.path.getsize(path) >= HEADER_BYTES:
            with open(path, "rb") as f:
                header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
            if header.size >= 3 and header[0] == FW_MAGIC:
                ntok = int(header[2])
                with open(path, "rb") as f:
                    f.seek(HEADER_BYTES)
                    file_size = os.path.getsize(path)
                    data_bytes = file_size - HEADER_BYTES
                    dtype = np.uint32 if data_bytes == ntok * 4 else np.uint16
                    tokens = np.frombuffer(f.read(), dtype=dtype)
                return np.asarray(tokens)
    except Exception:
        pass
    return np.memmap(path, dtype=np.uint16, mode='r')

def get_data_loader_streamshift(
    data_path: str,
    num_samples: int,
    batch_size: int,
    seq_length: int,
    device: str
):
    """
    Yields (x, y) pairs where y is x shifted by 1.
    Stride = seq_length + 1 to avoid overlap.
    """
    tokens = _load_data_shard(data_path)
    if isinstance(tokens, np.memmap):
        tokens = np.array(tokens, dtype=np.int64) # Load into RAM if memmap
    else:
        tokens = tokens.astype(np.int64)
    tokens = torch.from_numpy(tokens)

    # We need (seq_length + 1) tokens per sample
    needed_per_sample = seq_length + 1
    total_tokens = len(tokens)
    max_samples = total_tokens // needed_per_sample

    if num_samples == 0 or num_samples > max_samples:
        num_samples = max_samples

    used_tokens = num_samples * needed_per_sample
    data = tokens[:used_tokens].view(num_samples, needed_per_sample)

    for i in range(0, num_samples, batch_size):
        batch = data[i : i + batch_size].to(device) # [B, T+1]
        x = batch[:, :-1]
        y = batch[:, 1:]
        yield x, y


# ==============================================================================
# 4) Spectrum Calculation (Online Covariance + Gradient SVD)
# ==============================================================================


def _get_param_by_path(layer: nn.Module, path: str) -> torch.nn.Parameter:
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        if hasattr(obj, "requires_grad"):
            return obj
        raise TypeError(f"Resolved '{path}' is not a Parameter/Tensor")
    return obj

class OnlineCov:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, X: np.ndarray):
        # X: [B, dim]
        if X.size == 0: return
        n_new = X.shape[0]
        mean_new = X.mean(axis=0)
        delta = mean_new - self.mean
        total = self.n + n_new
        
        # M2 += X'X - n_new * outer(mean_new, mean_new) 
        # But clearer update for centered sum of squares:
        # term1 = sum( (x - mean_new)^2 )
        # term2 = term based on delta mean
        # Standard Welford/Chan update for covariance matrix M2:
        # M2_new = M2_old + (X - mean_new)^T (X - mean_new) + ... (cross term)
        # Actually simpler: M2 stores scatter matrix (unnormalized covariance * (n-1))
        
        # Batch update formula:
        # M2 += (X - mean_new).T @ (X - mean_new) + n * n_new / total * (delta.T @ delta)
        # However, typically easier to accumulate X^T X and sum(X).
        # But we want to avoid catastrophic cancellation. 
        # Let's use the provided logic which matches previous scripts.
        
        # self.M2 += (np.dot(X.T, X) - n_new * np.outer(mean_new, mean_new)) # This assumes X is not centered?
        # Let's stick to the implementation from previous robust scripts if possible.
        # Below is from 14_sa_truncated_rope_1gpu.py:
        self.M2 += (np.dot(X.T, X) - n_new * np.outer(mean_new, mean_new))
        if self.n > 0:
            self.M2 += (self.n * n_new / total) * np.outer(delta, delta)
        self.mean += (n_new / total) * delta
        self.n = total

    def covariance(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros((self.dim, self.dim))
        return self.M2 / (self.n - 1)


@torch.no_grad()
def compute_covariance_spectrum(
    model: GPT,
    data_loader,
    layer_idx: int,
    sliding_window_num_blocks: int,
    mask_block_size: int = 128
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    C = model.config.n_embd
    cov = OnlineCov(C)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, _ in data_loader:
            idx = idx.to(device)
            # Iterate over batch items (each is a 1D sequence), matching gradient path
            B_in = idx.size(0)
            for b in range(B_in):
                idx_b = idx[b]  # [T] — 1D
                # Embeddings & Mask
                x, x0, ve_all, block_mask = embed_inputs_and_mask_sparse(
                    model, idx_b, sliding_window_num_blocks, mask_block_size
                )
                # Forward to layer
                h_at = run_until_layer(model, x, x0, ve_all, block_mask, layer_idx)

                # Update covariance
                if torch.isfinite(h_at).all():
                    chunk = h_at.reshape(-1, C).float().cpu().numpy()
                    cov.update(chunk)

    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals


def get_layer_param(model: GPT, layer_idx: int, path: str) -> torch.nn.Parameter:
    # "attn.c_proj.weight" -> model.transformer.h[layer_idx].attn.c_proj.weight
    layer = model.transformer.h[layer_idx]
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def compute_gradient_svd_spectrum(
    model: GPT,
    data_loader,
    layer_idx: int,
    param_path: str,
    num_samples: int,
    sliding_window_num_blocks: int,
    mask_block_size: int = 128
):
    model.train() # Enable gradients
    device = next(model.parameters()).device
    
    target_param = get_layer_param(model, layer_idx, param_path)
    
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    target_param.requires_grad = True
    
    G_list = []
    collected = 0
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, targets in data_loader:
            if collected >= num_samples:
                break
            
            # Process one sample at a time to keep memory low
            B_in = idx.size(0)
            for b in range(B_in):
                if collected >= num_samples:
                    break
                
                x_b = idx[b:b+1].to(device)
                y_b = targets[b:b+1].to(device)
                
                model.zero_grad(set_to_none=True)
                logits, loss = model(x_b, y_b, sliding_window_num_blocks, mask_block_size)
                
                loss.backward()
                
                grad = target_param.grad.detach().flatten().float().cpu()
                G_list.append(grad)
                collected += 1

    if collected == 0:
        return np.array([])
    
    G = torch.stack(G_list) # [N, D]
    # SVD
    s = torch.linalg.svdvals(G)
    return s.numpy()


def analyze_power_law(spectrum: np.ndarray, tail_start: int, tail_finish: int) -> Dict[str, float]:
    spectrum = np.sort(spectrum)[::-1]
    i = np.arange(1, len(spectrum) + 1)
    
    # Safe indices
    start_idx = max(0, min(tail_start - 1, len(spectrum) - 2))
    end_idx = min(tail_finish, len(spectrum))
    if end_idx <= start_idx + 2:
        return {"alpha": np.nan, "r_squared": np.nan}

    y = spectrum[start_idx:end_idx]
    x = i[start_idx:end_idx]
    
    log_y = np.log(y + 1e-20)
    log_x = np.log(x)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    return {"alpha": -slope, "r_squared": r_value**2}


# ==============================================================================
# 5) Main
# ==============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: str) -> GPT:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Defaults
    args_dict = {
        "vocab_size": 50304,
        "n_layer": 12,
        "n_head": 6,
        "n_embd": 768,
        "sequence_length": 65536,
        "lm_head_softcap": 15.0
    }
    
    # Update from checkpoint args
    if isinstance(ckpt, dict) and "args" in ckpt:
        a = ckpt["args"]
        # Handle Namespace or dict
        if not isinstance(a, dict):
            a = vars(a)
        
        args_dict.update({k: v for k, v in a.items() if k in args_dict})
        if "lm_head_softcap" in a:
             args_dict["lm_head_softcap"] = a["lm_head_softcap"]

    cfg = GPTConfig(
        vocab_size=args_dict["vocab_size"],
        n_layer=args_dict["n_layer"],
        n_head=args_dict["n_head"],
        n_embd=args_dict["n_embd"],
        seq_length=args_dict["sequence_length"]
    )
    
    model = GPT(cfg, lm_head_softcap=args_dict["lm_head_softcap"])
    
    # Load state dict
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt: state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt: state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt: state_dict = ckpt["state_dict"]
    
    # Strip prefixes
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."): k = k[10:]
        if k.startswith("module."): k = k[7:]
        if k.startswith("model."): k = k[6:]
        
        # Filter optim/rng/etc
        if k.startswith(("optimizer", "rng", "scheduler")): continue
        new_sd[k] = v
        
    try:
        model.load_state_dict(new_sd, strict=True)
    except Exception as e:
        print(f"Strict load failed: {e}. Trying strict=False")
        model.load_state_dict(new_sd, strict=False)
        
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--validation_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_index", type=int, default=-1)
    parser.add_argument("--layers", type=str, default="auto")
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight", help="Comma-separated parameter paths for gradient analysis")
    
    # Analysis params
    parser.add_argument("--seq_length", type=int, default=65536)
    parser.add_argument("--num_samples_grad", type=int, default=100)
    parser.add_argument("--num_samples_cov", type=int, default=500)
    parser.add_argument("--cov_batch_size", type=int, default=4)
    parser.add_argument("--grad_batch_size", type=int, default=4)
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)
    
    # Defaults via standard args fallback, but CLI can override
    parser.add_argument("--lm_head_softcap", type=float, default=15.0)
    parser.add_argument("--block_size", type=int, default=128)
    
    # Window schedule (Dynamic)
    parser.add_argument("--window_warmup_steps", type=int, default=2500)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Files
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_ckpts = sorted(list(ckpt_dir.glob("ckpt_step_*.pt")), key=lambda p: str(p))
    # Filter/sort by step
    def get_step(p):
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else 999999
    all_ckpts = sorted(all_ckpts, key=get_step)
    
    if not all_ckpts:
        print(f"No checkpoints in {ckpt_dir}")
        return

    if args.task_index >= 0:
        if not 0 <= args.task_index < len(all_ckpts):
            raise IndexError(f"--task_index {args.task_index} out of bounds (n={len(all_ckpts)})")
        ckpts_to_run = [all_ckpts[args.task_index]]
    else:
        ckpts_to_run = all_ckpts
        
    print(f"Processing {len(ckpts_to_run)} checkpoints.")
    
    # Parse layers
    if args.layers == "all":
        layers_list = list(range(12)) # provisional
    else:
        layers_list = [int(x) for x in args.layers.split(",")]
        
    matrices_to_analyze = [m.strip() for m in args.matrices.split(",")]
    for cp_path in tqdm(ckpts_to_run):
        try:
            step = get_step(cp_path)
            
            # --- Dynamic Window Schedule Calculation ---
            # wfrac = min(step / warmup, 1.0)
            # window_tokens_raw = 1728.0 * wfrac
            # sw_blocks = ceil(raw / 128), clamped [1, 14]

            warmup = max(1, args.window_warmup_steps)
            wfrac = min(step / warmup, 1.0)
            window_tokens_raw = 1728.0 * wfrac
            sw_blocks_val = int(math.ceil(window_tokens_raw / args.block_size))
            sw_blocks_val = max(1, min(sw_blocks_val, 14))

            print(f"Step {step}: sw_blocks={sw_blocks_val}")

            # Check if all files exist (OUTSIDE layer loop)
            all_exist = True
            for layer_idx in layers_list:
                if layer_idx >= 12:
                    continue
                cov_path = out_dir / f"cov_spectrum_{step:07d}_L{layer_idx:02d}.npy"
                for matrix_name in matrices_to_analyze:
                    matrix_name_clean = matrix_name.replace('.', '_')
                    grad_path = out_dir / f"grad_spectrum_{step:07d}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                    weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                    if not (cov_path.exists() and grad_path.exists() and weight_path.exists()):
                        all_exist = False
                        break
                if not all_exist:
                    break

            if all_exist:
                print(f"[info] All outputs exist for step {step}; skipping.")
                continue

            # Load model (OUTSIDE layer loop)
            model = load_model_from_checkpoint(str(cp_path), device)

            # Set softcap if provided
            if args.lm_head_softcap is not None:
                model.lm_head_softcap = args.lm_head_softcap

            checkpoint_results = []

            for layer_idx in layers_list:
                if layer_idx >= model.config.n_layer:
                    continue

                # Covariance
                cov_path = out_dir / f"cov_spectrum_{step:07d}_L{layer_idx:02d}.npy"
                if not cov_path.exists():
                    dl = get_data_loader_streamshift(args.validation_data_path, args.num_samples_cov, args.cov_batch_size, args.seq_length, device)
                    cov_s = compute_covariance_spectrum(model, dl, layer_idx, sw_blocks_val, args.block_size)
                    np.save(cov_path, cov_s)
                else:
                    cov_s = np.load(cov_path)

                cov_res = analyze_power_law(cov_s, args.tail_start, args.tail_finish)

                # Loop over matrices for gradient
                for matrix_name in matrices_to_analyze:
                    matrix_name_clean = matrix_name.replace('.', '_')
                    grad_path = out_dir / f"grad_spectrum_{step:07d}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                    weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))

                    if not grad_path.exists():
                        dl = get_data_loader_streamshift(args.validation_data_path, args.num_samples_grad, args.grad_batch_size, args.seq_length, device)
                        grad_s = compute_gradient_svd_spectrum(
                            model, dl, layer_idx, matrix_name, args.num_samples_grad, sw_blocks_val, args.block_size
                        )
                        np.save(grad_path, grad_s)
                    else:
                        grad_s = np.load(grad_path)

                    grad_res = analyze_power_law(grad_s, args.tail_start, args.tail_finish)

                    weight_s = None
                    if weight_path.exists():
                        weight_s = np.load(weight_path)
                    else:
                        try:
                            weight_theta = _get_param_by_path(model.transformer.h[layer_idx], matrix_name)
                            weight_s = torch.linalg.svdvals(weight_theta.detach().float()).cpu().numpy()
                            np.save(weight_path, weight_s)
                        except Exception as e:
                            weight_s = None
                    if weight_s is None:
                        weight_fit = {'alpha': float('nan'), 'r_squared': float('nan')}
                    else:
                        weight_fit = analyze_power_law(weight_s, args.tail_start, args.tail_finish)

                    checkpoint_results.append({
                        "step": step,
                        "layer": layer_idx,
                        "matrix": matrix_name,
                        "cov_alpha": cov_res["alpha"],
                        "cov_r2": cov_res["r_squared"],
                        "grad_alpha": grad_res["alpha"],
                        "grad_r2": grad_res["r_squared"],
                        'weight_alpha': float(weight_fit['alpha']),
                        'weight_r2': float(weight_fit['r_squared']),
                    })

            # Save summary with append logic
            if checkpoint_results:
                csv_path = out_dir / f"summary_step_{step:07d}.csv"
                new_df = pd.DataFrame(checkpoint_results)
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
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error on {cp_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
