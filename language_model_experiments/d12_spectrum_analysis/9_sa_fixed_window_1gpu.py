# sandbox:/mnt/data/9_sa_fixed_window_1gpu_fixed.py
import os
import re
import math
import pathlib
import argparse
import time
import gc
from dataclasses import dataclass
from typing import Optional, Tuple, List
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import tqdm

# --- Flex Attention (Fixed Window) ---
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention, dynamic=False)
    create_block_mask = torch.compile(create_block_mask, dynamic=False)
except Exception:
    flex_attention = None
    create_block_mask = None

# =============================================================================
# Model (must match 9_fixed_window.py)
# =============================================================================

RMS_EPS = 1e-6
EOS_ID = 50256
WINDOW_SIZE = 1024  # fixed window used in 9_fixed_window.py

def apply_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=RMS_EPS)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        key = (x.device, x.dtype, seq_len)
        if key != self._cache_key:
            self._cache_key = key
            inv_freq = self.inv_freq.to(device=x.device)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos().to(dtype=x.dtype)
            sin = freqs.sin().to(dtype=x.dtype)
            self.cos_cached = cos
            self.sin_cached = sin
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    assert x.ndim == 4
    x1, x2 = x.chunk(2, dim=3)  # Use chunk for slight speed boost
    y1 = x1 * cos - x2 * sin   # CORRECT formula
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, v1, block_mask):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flex_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)

        return y, v1


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor, v1: Optional[torch.Tensor], x0: torch.Tensor, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(apply_norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(apply_norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = WINDOW_SIZE 
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        idx = idx.view(-1)
        if targets is not None:
            targets = targets.view(-1)

        x, x0, block_mask = embed_inputs_and_mask_1d(self, idx)
        v1 = None

        skips = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = apply_norm(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        logits = logits.float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# =============================================================================
# Data loading (must match next-token objective used in training)
# =============================================================================

HEADER_BYTES = 256 * 4
FW_MAGIC = 20240520

def _load_data_shard(path: str) -> np.ndarray:
    # Copied pattern from your other scripts: supports FineWeb header format.
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
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_data_loader_streamshift(
    data_path: str,
    num_samples: int,
    batch_size: int,
    seq_length: int,
    device: str,
):
    """
    Produces batches of (x, y) with y being the true next-token shift of x,
    matching the training objective for 9_fixed_window.py.
    """
    raw = _load_data_shard(data_path)
    tokens = torch.from_numpy(np.asarray(raw, dtype=np.int64))

    stride = seq_length + 1
    num_sequences = len(tokens) // stride
    if num_samples <= 0 or num_samples > num_sequences:
        num_samples = num_sequences

    data = tokens[: num_samples * stride].view(num_samples, stride)
    for i in range(0, num_samples, batch_size):
        batch = data[i: i + batch_size]
        x = batch[:, :-1].to(device, non_blocking=True)
        y = batch[:, 1:].to(device, non_blocking=True)
        yield x, y

# =============================================================================
# Flex mask + embedding helpers (operate on 1D idx)
# =============================================================================

@torch.no_grad()
def build_block_mask_from_idx_1d(idx_1d: torch.Tensor):
    assert create_block_mask is not None, "flex_attention/create_block_mask not available."
    docs = (idx_1d == EOS_ID).cumsum(0)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_doc = docs[q_idx] == docs[kv_idx]
        in_window = (q_idx - kv_idx) < WINDOW_SIZE
        return causal & same_doc & in_window

    T = idx_1d.numel()
    return create_block_mask(document_causal_mask, None, None, T, T, device=idx_1d.device, _compile=True)

@torch.no_grad()
def embed_inputs_and_mask_1d(model: GPT, idx_1d: torch.Tensor):
    x = model.transformer.wte(idx_1d[None])
    x = apply_norm(x)
    x0 = x
    block_mask = build_block_mask_from_idx_1d(idx_1d)
    return x, x0, block_mask

# =============================================================================
# Probes
# =============================================================================

@torch.no_grad()
def run_until_layer_unet(model: GPT, idx_1d: torch.Tensor, layer_idx: int) -> torch.Tensor:
    x, x0, block_mask = embed_inputs_and_mask_1d(model, idx_1d)
    v1 = None

    E = model.num_encoder_layers
    D = model.num_decoder_layers

    cur = x
    skips = []

    for i in range(E):
        cur, v1 = model.transformer.h[i](cur, v1, x0, block_mask)
        skips.append(cur)
        if i == layer_idx:
            return apply_norm(cur)

    for i in range(D):
        skip = skips.pop()
        cur = cur + model.skip_weights[i] * skip
        cur, v1 = model.transformer.h[E + i](cur, v1, x0, block_mask)
        if (E + i) == layer_idx:
            return apply_norm(cur)

    raise RuntimeError(f"layer_idx out of range: {layer_idx}")

def _get_param_by_path(layer: nn.Module, path: str) -> torch.nn.Parameter:
    """Retrieve a parameter from a module by dotted path."""
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        if hasattr(obj, "requires_grad"):
            return obj
        raise TypeError(f"Resolved '{path}' is not a Parameter/Tensor")
    return obj


def analyze_power_law(spectrum: np.ndarray, tail_start: int, tail_finish: int) -> dict:
    spectrum = np.sort(spectrum)[::-1]
    i = np.arange(1, len(spectrum) + 1)
    start_idx = min(tail_start - 1, max(0, len(spectrum) - 2))
    end_idx = min(tail_finish, len(spectrum)) if tail_finish > 0 else len(spectrum)
    tail_spectrum, tail_i = spectrum[start_idx:end_idx], i[start_idx:end_idx]

    results = {"alpha": np.nan, "r_squared": np.nan}
    if len(tail_spectrum) > 2:
        log_tail_spectrum = np.log(tail_spectrum)
        finite_mask = np.isfinite(log_tail_spectrum)
        if finite_mask.sum() >= 2:
            slope, _, r_value, _, _ = linregress(np.log(tail_i[finite_mask]), log_tail_spectrum[finite_mask])
            results.update({"alpha": -slope, "r_squared": r_value**2})
    return results

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

    def covariance(self):
        if self.n < 2:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        return self.M2 / (self.n - 1)

@torch.no_grad()
def compute_covariance_spectrum(model: GPT, data_loader, layer_idx: int, autocast_ctx) -> np.ndarray:
    model.eval()
    C = model.config.n_embd
    cov = OnlineCov(C)

    for x_batch, _ in data_loader:
        B = x_batch.size(0)
        for b in range(B):
            idx_1d = x_batch[b].contiguous()
            with autocast_ctx:
                h = run_until_layer_unet(model, idx_1d, layer_idx)
            if not torch.isfinite(h).all():
                continue
            cov.update(h.reshape(-1, C).float().cpu().numpy())

    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals

def get_layer_param(model: GPT, layer_idx: int, path: str) -> torch.nn.Parameter:
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
    autocast_ctx,
    max_samples_for_memory: int = 512,
) -> np.ndarray:
    model.train()
    theta = get_layer_param(model, layer_idx, param_path)

    if num_samples > max_samples_for_memory:
        print(f"[warn] num_samples_grad={num_samples} can be very memory-heavy for {param_path}.")

    for p in model.parameters():
        p.requires_grad_(False)
    theta.requires_grad_(True)

    P = theta.numel()
    G = torch.zeros((num_samples, P), dtype=torch.float32, device="cpu")

    collected = 0
    for x_batch, y_batch in data_loader:
        if collected >= num_samples:
            break
        B = x_batch.size(0)
        for b in range(B):
            if collected >= num_samples:
                break
            idx_1d = x_batch[b].contiguous()
            tgt_1d = y_batch[b].contiguous()

            with autocast_ctx:
                _, loss = model(idx_1d, tgt_1d)
            model.zero_grad(set_to_none=True)
            loss.backward()

            grad_flat = theta.grad.detach().reshape(-1).to("cpu", non_blocking=True)
            G[collected, :] = grad_flat
            collected += 1

    if collected == 0:
        return np.array([], dtype=np.float64)

    with torch.no_grad():
        s = torch.linalg.svdvals(G[:collected].float())
    return s.cpu().numpy()

# =============================================================================
# Checkpoint loading (9_fixed_window.py saves under key="model")
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: str, block_size_override: Optional[int] = None) -> GPT:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = GPTConfig(
        vocab_size=50304,
        block_size=WINDOW_SIZE if block_size_override is None else int(block_size_override),
        n_layer=12,
        n_head=6,
        n_embd=768
    )
    model = GPT(cfg)

    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    if state_dict is None and isinstance(ckpt, dict):
        only_tensors = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        state_dict = only_tensors if only_tensors else ckpt

    def strip_prefix(k: str) -> str:
        for p in ("_orig_mod.", "module.", "model."):
            if k.startswith(p):
                return k[len(p):]
        return k

    cleaned = {strip_prefix(k): v for k, v in state_dict.items()
               if not k.startswith(("optimizer", "optimizers", "rng", "scheduler"))}

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as e:
        print(f"Strict load failed: {e}. Trying strict=False")
        model.load_state_dict(cleaned, strict=False)

    del ckpt, state_dict, cleaned
    gc.collect()

    model.to(device)
    model.eval()
    return model

def parse_layers(layers_str: str, total_layers: int) -> list[int]:
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

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Fixed Window Spectrum Analysis (safe batching, stream-shift targets)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--validation_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight")
    parser.add_argument("--task_index", type=int, default=-1)

    parser.add_argument("--seq_length", type=int, default=32768)
    parser.add_argument("--num_samples_grad", type=int, default=512)
    parser.add_argument("--num_samples_cov", type=int, default=512)
    parser.add_argument("--cov_batch_size", type=int, default=2)
    parser.add_argument("--grad_batch_size", type=int, default=2)
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ptdtype = torch.bfloat16 # Default to bfloat16 to match 10_flex_window
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    print(f"Using autocast: {device_type} {ptdtype}")

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = sorted(
        ckpt_dir.glob("checkpoint_step*.pt"),
        key=lambda p: int((re.search(r"(?:state|checkpoint)_step(\d+)\.pt$", p.name)
                           or re.search(r"(\d+)", p.stem)).group(1))
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if args.task_index != -1:
        if not 0 <= args.task_index < len(ckpt_paths):
            raise IndexError(f"--task_index {args.task_index} out of bounds (n={len(ckpt_paths)})")
        ckpt_paths = [ckpt_paths[args.task_index]]

    temp_model = load_model_from_checkpoint(str(ckpt_paths[0]), device="cpu", block_size_override=args.seq_length)
    layers_to_analyze = parse_layers(args.layers, temp_model.config.n_layer)
    del temp_model
    gc.collect()

    matrices_to_analyze = [m.strip() for m in args.matrices.split(",")]
    for path in tqdm(ckpt_paths, desc="Checkpoints"):
        m = (re.search(r"(?:state|checkpoint)_step(\d+)", path.name) or re.search(r"(\d+)", path.stem))
        step_str = m.group(1) if m else "0"
        step = int(step_str)

        # Check if all files exist (outside layer loop)
        all_files_exist = True
        for layer_idx in layers_to_analyze:
            cov_path = output_dir / f"cov_spectrum_{step_str}_L{layer_idx:02d}.npy"
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                grad_path = output_dir / f"grad_spectrum_{step_str}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                if not (cov_path.exists() and grad_path.exists() and weight_path.exists()):
                    all_files_exist = False
                    break
            if not all_files_exist:
                break

        if all_files_exist:
            print(f"All files for Step {step} exist. Skipping.")
            continue

        # Load model only if needed
        model = load_model_from_checkpoint(str(path), device=device, block_size_override=args.seq_length)

        results = []
        for layer_idx in layers_to_analyze:
            # Covariance spectrum (once per layer)
            cov_path = output_dir / f"cov_spectrum_{step_str}_L{layer_idx:02d}.npy"
            if cov_path.exists():
                cov_eigvals = np.load(cov_path)
            else:
                dl_cov = get_data_loader_streamshift(
                    args.validation_data_path, args.num_samples_cov, args.cov_batch_size, args.seq_length, device
                )
                cov_eigvals = compute_covariance_spectrum(model, dl_cov, layer_idx, autocast_ctx)
                np.save(cov_path, cov_eigvals)

            cov_fit = analyze_power_law(cov_eigvals, args.tail_start, args.tail_finish)

            # Loop over matrices for gradient analysis
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                grad_path = output_dir / f"grad_spectrum_{step_str}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                if grad_path.exists():
                    grad_svd = np.load(grad_path)
                else:
                    dl_grad = get_data_loader_streamshift(
                        args.validation_data_path, args.num_samples_grad, args.grad_batch_size, args.seq_length, device
                    )
                    grad_svd = compute_gradient_svd_spectrum(model, dl_grad, layer_idx, matrix_name, args.num_samples_grad, autocast_ctx)
                    np.save(grad_path, grad_svd)

                grad_fit = analyze_power_law(grad_svd, args.tail_start, args.tail_finish)

                # Weight spectrum
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

                results.append({
                    "step": step,
                    "layer": layer_idx,
                    "matrix": matrix_name,
                    "seq_length": int(args.seq_length),
                    'weight_alpha': float(weight_fit['alpha']),
                    'weight_r2': float(weight_fit['r_squared']),
                    "cov_alpha": float(cov_fit["alpha"]),
                    "cov_r2": float(cov_fit["r_squared"]),
                    "grad_alpha": float(grad_fit["alpha"]),
                    "grad_r2": float(grad_fit["r_squared"]),
                })

        if results:
            csv_path = output_dir / f"summary_step_{step_str}.csv"
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
            print(f"Wrote {csv_path.name}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
