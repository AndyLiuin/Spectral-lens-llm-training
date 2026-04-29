# 11_sa_vte_1gpu_FIXED.py
import os
import re
import gc
import json
import math
import pathlib
import argparse
from typing import Optional
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import tqdm
from scale_arch_utils import parse_layers_spec, resolve_model_dims, validate_dims

# --- Flex Attention ---
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention, dynamic=False)
    create_block_mask = torch.compile(create_block_mask, dynamic=False)
except Exception as e:
    flex_attention = None
    create_block_mask = None
    print(f"Warning: flex_attention not available: {e}")

EOS_ID = 50256

# ==============================================================================
# --- 1. Model Definitions (VTE + Flex Attention + UNet) ---
# ==============================================================================

def apply_norm(x):
    return F.rms_norm(x, (x.size(-1),))

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
    x1, x2 = x.chunk(2, dim=3)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
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

    def forward(self, x, vi, block_mask):
        B, T, _ = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        
        # VTE mixing
        if vi is not None:
            v = ((1 - self.lamb) * v + self.lamb * vi.view_as(v)).to(v.dtype)
        
        cos, sin = self.rotary(q)
        q = apply_norm(q)
        k = apply_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flex_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y, None

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x_attn, _ = self.attn(apply_norm(x), vi, block_mask)
        x = x + x_attn
        x = x + self.mlp(apply_norm(x))
        return x, None

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    block_size: int = 1024

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer

        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
                
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            vte = nn.Embedding(config.vocab_size, config.n_embd * config.n_layer),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, attn_blocksize: int = 1024):
        # Flatten
        if idx.ndim == 2: idx = idx.view(-1)
        if targets is not None and targets.ndim == 2: targets = targets.view(-1)

        x, x0, vi_chunks, block_mask = embed_inputs_and_mask_vte(self, idx, attn_blocksize)
        v1 = None # Unused in loop, but signature consistency

        skips = []
        for i in range(self.num_encoder_layers):
            x, _ = self.transformer.h[i](x, vi_chunks[i], x0, block_mask)
            skips.append(x)
        
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            lid = self.num_encoder_layers + i
            x, _ = self.transformer.h[lid](x, vi_chunks[lid], x0, block_mask)
        
        x = apply_norm(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        logits = logits.float()

        loss = None
        if targets is not None:
             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==============================================================================
# --- 2. Helper Functions ---
# ==============================================================================

@torch.no_grad()
def build_block_mask_from_idx(idx_1d: torch.Tensor, attn_blocksize: int):
    assert create_block_mask is not None
    docs = (idx_1d == EOS_ID).cumsum(0)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_doc = docs[q_idx] == docs[kv_idx]
        in_window = (q_idx - kv_idx) < attn_blocksize
        return causal & same_doc & in_window

    T = idx_1d.numel()
    return create_block_mask(document_causal_mask, None, None, T, T, device=idx_1d.device, _compile=True)

@torch.no_grad()
def embed_inputs_and_mask_vte(model: GPT, idx_1d: torch.Tensor, attn_blocksize: int):
    x = model.transformer.wte(idx_1d[None])
    x = apply_norm(x)
    x0 = x
    vi_chunks = model.transformer.vte(idx_1d[None]).chunk(model.config.n_layer, dim=-1)
    
    block_mask = build_block_mask_from_idx(idx_1d, attn_blocksize)
    return x, x0, vi_chunks, block_mask

# =============================================================================
# Spectrum Probe
# =============================================================================

@torch.no_grad()
def run_until_layer_vte(model: GPT, idx_1d: torch.Tensor, layer_idx: int, attn_blocksize: int):
    x, x0, vi_chunks, block_mask = embed_inputs_and_mask_vte(model, idx_1d, attn_blocksize)

    E = model.num_encoder_layers
    D = model.num_decoder_layers
    
    # Encoder
    skips = []
    for i in range(E):
        x, _ = model.transformer.h[i](x, vi_chunks[i], x0, block_mask)
        skips.append(x)
        if i == layer_idx: return apply_norm(x)

    # Decoder
    for i in range(D):
        x = x + model.skip_weights[i] * skips.pop()
        lid = E + i
        x, _ = model.transformer.h[lid](x, vi_chunks[lid], x0, block_mask)
        if lid == layer_idx: return apply_norm(x)
        
    raise RuntimeError(f"layer_idx {layer_idx} out of range")

class OnlineCov:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, X: np.ndarray):
        if X.size == 0: return
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
        if self.n < 2: return np.zeros((self.dim, self.dim))
        return self.M2 / (self.n - 1)

@torch.no_grad()
def compute_covariance_spectrum(model, data_loader, layer_idx: int, attn_blocksize: int, autocast_ctx):
    model.eval()
    cov = OnlineCov(model.config.n_embd)
    
    for x_batch, _ in data_loader:
        B = x_batch.size(0)
        for b in range(B):
            idx_1d = x_batch[b].contiguous()
            with autocast_ctx:
                x_at = run_until_layer_vte(model, idx_1d, layer_idx, attn_blocksize)
            if not torch.isfinite(x_at).all(): continue
            cov.update(x_at.reshape(-1, model.config.n_embd).float().cpu().numpy())
            
    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals

def compute_gradient_svd_spectrum(
    model: GPT, 
    data_loader, 
    layer_idx: int, 
    param_path: str, 
    num_samples: int, 
    attn_blocksize: int,
    autocast_ctx
):
    model.train()
    
    # Resolve parameter
    obj = model.transformer.h[layer_idx]
    for part in param_path.split("."):
        obj = getattr(obj, part)
    theta = obj
    
    for p in model.parameters(): p.requires_grad_(False)
    if isinstance(theta, torch.nn.Parameter):
        theta.requires_grad_(True)

    P = theta.numel()
    G = torch.zeros((num_samples, P), dtype=torch.float32, device='cpu')

    collected = 0
    for x_batch, y_batch in data_loader:
        if collected >= num_samples: break
        B = x_batch.size(0)
        for b in range(B):
            if collected >= num_samples: break
            idx_1d = x_batch[b].contiguous()
            tgt_1d = y_batch[b].contiguous()
            
            model.zero_grad(set_to_none=True)
            with autocast_ctx:
                _, loss = model(idx_1d, tgt_1d, attn_blocksize=attn_blocksize)
            
            if loss is None or not torch.isfinite(loss): continue
            loss.backward()
            
            grad_flat = theta.grad.detach().reshape(-1).to('cpu', non_blocking=True)
            G[collected, :] = grad_flat
            collected += 1

    if collected == 0: return np.array([], dtype=np.float64)

    with torch.no_grad():
        s = torch.linalg.svdvals(G[:collected].float())
    return s.cpu().numpy()

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
    spectrum = np.asarray(spectrum)
    spectrum = spectrum[np.isfinite(spectrum) & (spectrum > 0)]
    if spectrum.size == 0: return {"alpha": np.nan, "r_squared": np.nan}

    spectrum = np.sort(spectrum)[::-1]
    i = np.arange(1, len(spectrum) + 1)

    s_idx = max(0, min(len(spectrum)-2, tail_start-1))
    e_idx = min(len(spectrum), tail_finish) if tail_finish > 0 else len(spectrum)

    y = spectrum[s_idx:e_idx]
    x = i[s_idx:e_idx]

    if len(y) > 2:
        logy = np.log(y)
        mask = np.isfinite(logy)
        if mask.sum() >= 2:
            slope, _, r_val, _, _ = linregress(np.log(x[mask]), logy[mask])
            return {"alpha": -slope, "r_squared": r_val**2}
    return {"alpha": np.nan, "r_squared": np.nan}

# =============================================================================
# Data Loading
# =============================================================================

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
    return np.memmap(path, dtype=np.uint16, mode='r')

def get_data_loader_streamshift(data_path: str, num_samples: int, batch_size: int, seq_length: int, device: str):
    raw = _load_data_shard(data_path)
    tokens = torch.from_numpy(np.asarray(raw, dtype=np.int64))
    stride = seq_length + 1
    num_sequences = len(tokens) // stride
    if num_samples <= 0 or num_samples > num_sequences:
        num_samples = num_sequences
    
    data = tokens[:num_samples*stride].view(num_samples, stride)
    for i in range(0, num_samples, batch_size):
        batch = data[i:i+batch_size]
        x = batch[:, :-1].to(device, non_blocking=True)
        y = batch[:, 1:].to(device, non_blocking=True)
        yield x, y

# =============================================================================
# Main
# =============================================================================

def _strip_prefix(k: str) -> str:
    for p in ("_orig_mod.", "module.", "model."):
        if k.startswith(p): return k[len(p):]
    return k

def load_training_args(ckpt_dir: pathlib.Path) -> dict:
    candidates = [ckpt_dir / "args.json", ckpt_dir.parent / "args.json"]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f: return json.load(f)
            except: pass
    return {}

def calculate_dynamic_window(step: int, wmin: int, wmax: int, wwarm: int) -> int:
    if wwarm <= 0: return int(wmax)
    frac = min(step / wwarm, 1.0)
    val = wmin + frac * (wmax - wmin)
    snapped = 64 * int(val // 64)
    return int(max(wmin, min(wmax, snapped)))

def load_model_from_checkpoint(checkpoint_path: str, device: str, cfg: GPTConfig) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GPT(cfg)
    
    state_dict = None
    if isinstance(ckpt, dict):
        if "model" in ckpt: state_dict = ckpt["model"]
        elif "state_dict" in ckpt: state_dict = ckpt["state_dict"]
    if state_dict is None: state_dict = ckpt if isinstance(ckpt, dict) else None
    
    if state_dict is None: raise RuntimeError("No state dict found")
    
    cleaned = {_strip_prefix(k): v for k, v in state_dict.items() if torch.is_tensor(v)}
    
    # Strict load (required)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) or len(unexpected):
        # Try strict=True to get a full error message
        try:
            model.load_state_dict(cleaned, strict=True)
        except Exception:
            pass
        raise RuntimeError(
            f"STRICT LOAD FAILED for {checkpoint_path}\n"
            f"Missing ({len(missing)}): {missing[:5]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:5]}"
        )
        
    model.to(device).eval()
    return model

def parse_layers(layers_str: str, total_layers: int) -> list[int]:
    return parse_layers_spec(layers_str, total_layers)

def main():
    parser = argparse.ArgumentParser(description="Spectrum Analysis (VTE + Flex Window Fixed)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--validation_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="auto")
    parser.add_argument("--model_profile", type=str, default="d12", choices=["d12", "d24", "d36", "d48"])
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight")
    parser.add_argument("--task_index", type=int, default=-1)
    
    parser.add_argument("--seq_length", type=int, default=32768)
    parser.add_argument("--num_samples_grad", type=int, default=512)
    parser.add_argument("--num_samples_cov", type=int, default=512)
    parser.add_argument("--cov_batch_size", type=int, default=4)
    parser.add_argument("--grad_batch_size", type=int, default=4)
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)
    
    # Window params (fallback)
    parser.add_argument("--window_min", type=int, default=64)
    parser.add_argument("--window_max", type=int, default=1792)
    parser.add_argument("--window_warmup_steps", type=int, default=4000)
    
    args = parser.parse_args()
    
    if flex_attention is None: raise RuntimeError("Requires flex_attention")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load args.json
    train_args = load_training_args(ckpt_dir)
    wmin = int(train_args.get("window_min", args.window_min))
    wmax = int(train_args.get("window_max", args.window_max))
    wwarm = int(train_args.get("window_warmup_steps", args.window_warmup_steps))
    print(f"Window Schedule: min={wmin}, max={wmax}, warmup={wwarm}")
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[str(train_args.get("dtype", "bfloat16"))]
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type=="cuda" else nullcontext()

    prof_layer, prof_head, prof_embd = resolve_model_dims(
        argparse.Namespace(model_profile=args.model_profile, n_layer=None, n_head=None, n_embd=None),
        default=(12, 6, 768),
    )
    n_layer = int(args.n_layer) if args.n_layer is not None else int(train_args.get("n_layer", prof_layer))
    n_head = int(args.n_head) if args.n_head is not None else int(train_args.get("n_head", prof_head))
    n_embd = int(args.n_embd) if args.n_embd is not None else int(train_args.get("n_embd", prof_embd))
    validate_dims(n_layer, n_head, n_embd, require_even_layers=True)
    cfg = GPTConfig(
        vocab_size=int(train_args.get("vocab_size", 50304)),
        block_size=int(args.seq_length),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    
    ckpt_paths = sorted(
        list(ckpt_dir.glob("checkpoint_step*.pt")) + list(ckpt_dir.glob("ckpt_step_*.pt")) + list(ckpt_dir.glob("checkpoint_target*.pt")),
        key=lambda p: int((re.search(r"step_?(\d+)", p.stem) or re.search(r"(\d+)", p.stem)).group(1))
    )
    
    if not ckpt_paths: raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    if args.task_index != -1:
        if not 0 <= args.task_index < len(ckpt_paths):
            raise IndexError(f"--task_index {args.task_index} out of bounds (n={len(ckpt_paths)})")
        ckpt_paths = [ckpt_paths[args.task_index]]
        
    matrices_to_analyze = [m.strip() for m in args.matrices.split(",")]
    layers_to_analyze = parse_layers(args.layers, cfg.n_layer)
    
    for path in tqdm(ckpt_paths, desc="Checkpoints"):
        step = int((re.search(r"step_?(\d+)", path.stem) or re.search(r"(\d+)", path.stem)).group(1))
        attn_blocksize = calculate_dynamic_window(step, wmin, wmax, wwarm)
        print(f"\nStep {step}: attn_blocksize={attn_blocksize}")

        # Check if all files exist (outside layer loop)
        all_files_exist = True
        for layer_idx in layers_to_analyze:
            cp = out_dir / f"cov_spectrum_{step}_L{layer_idx:02d}.npy"
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                gp = out_dir / f"grad_spectrum_{step}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                weight_path = gp.with_name(gp.name.replace("grad_spectrum_", "weight_spectrum_"))
                if not (cp.exists() and gp.exists() and weight_path.exists()):
                    all_files_exist = False
                    break
            if not all_files_exist:
                break

        if all_files_exist:
            print(f"All files for Step {step} exist. Skipping.")
            continue

        # Load model only if needed
        model = load_model_from_checkpoint(str(path), device, cfg=cfg)
        results = []

        for layer_idx in layers_to_analyze:
            # Covariance spectrum (once per layer)
            cp = out_dir / f"cov_spectrum_{step}_L{layer_idx:02d}.npy"
            if cp.exists():
                cov_eigs = np.load(cp)
            else:
                dl = get_data_loader_streamshift(args.validation_data_path, args.num_samples_cov, args.cov_batch_size, args.seq_length, device)
                cov_eigs = compute_covariance_spectrum(model, dl, layer_idx, attn_blocksize, autocast_ctx)
                np.save(cp, cov_eigs)

            cov_fit = analyze_power_law(cov_eigs, args.tail_start, args.tail_finish)

            # Loop over matrices for gradient analysis
            for matrix_name in matrices_to_analyze:
                matrix_name_clean = matrix_name.replace('.', '_')
                gp = out_dir / f"grad_spectrum_{step}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                weight_path = gp.with_name(gp.name.replace("grad_spectrum_", "weight_spectrum_"))
                if gp.exists():
                    grad_s = np.load(gp)
                else:
                    dl = get_data_loader_streamshift(args.validation_data_path, args.num_samples_grad, args.grad_batch_size, args.seq_length, device)
                    grad_s = compute_gradient_svd_spectrum(model, dl, layer_idx, matrix_name, args.num_samples_grad, attn_blocksize, autocast_ctx)
                    np.save(gp, grad_s)

                grad_fit = analyze_power_law(grad_s, args.tail_start, args.tail_finish)

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
                    "attn_blocksize": attn_blocksize,
                    'weight_alpha': float(weight_fit['alpha']),
                    'weight_r2': float(weight_fit['r_squared']),
                    "cov_alpha": float(cov_fit["alpha"]),
                    "cov_r2": float(cov_fit["r_squared"]),
                    "grad_alpha": float(grad_fit["alpha"]),
                    "grad_r2": float(grad_fit["r_squared"])
                })

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
                except Exception:
                    fallback_path = csv_path.with_name(csv_path.stem + "_new.csv")
                    new_df.to_csv(fallback_path, index=False)
            else:
                new_df.to_csv(csv_path, index=False)
        del model
        gc.collect()
        if device_type == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
