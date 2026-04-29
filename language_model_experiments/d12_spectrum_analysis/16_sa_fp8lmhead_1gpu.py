import os
import re
import math
import pathlib
import argparse
import time
import gc
from dataclasses import dataclass
from typing import Tuple, Literal, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import tqdm

# Import Flex Attention
try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
    flex_attention = torch.compile(flex_attention, dynamic=False)
except ImportError:
    print("Warning: torch.nn.attention.flex_attention not found. This script requires PyTorch nightly/2.5+")
    flex_attention = None
    BlockMask = None

# =============================================================================
# Model Definitions (Matches 16_fp8lmhead.py but uses standard Linear in SA)
# =============================================================================

def apply_norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
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
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        
        self.rotary = Rotary(self.head_dim, max_seq_len=config.seq_len)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x, ve, block_mask):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        
        if ve is None:
            v = self.mix[0] * v
        else:
            if ve.ndim == 2:
                ve = ve[None]
            ve = ve.view(B, T, self.n_head, self.head_dim)
            v = self.mix[0] * v + self.mix[1] * ve.to(dtype=v.dtype)

        q, k = apply_norm(q), apply_norm(k)
        q = self.rotary(q)
        k = self.rotary(k)

        y = flex_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int, skip_attn_layer: Optional[int] = 7):
        super().__init__()
        self.layer_idx = layer_idx
        self.skip_attn_layer = skip_attn_layer
        self.attn = CausalSelfAttention(config) if layer_idx != skip_attn_layer else None
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
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

    def forward(self, tokens_1d: torch.Tensor):
        ve0 = self.emb[0](tokens_1d) 
        ve1 = self.emb[1](tokens_1d)
        ve2 = self.emb[2](tokens_1d)
        pattern = [ve0, ve1, ve2, None, None, None, None, None, None, ve0, ve1, ve2]
        assert len(pattern) == self.n_layer
        return pattern

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 128
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    seq_len: int = 65536


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, lm_head_softcap: float = 15.0):
        super().__init__()
        self.config = config
        self.lm_head_softcap = float(lm_head_softcap)
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, i, skip_attn_layer=7) for i in range(config.n_layer)]),
            )
        )
        self.value_embeds = ValueTokenEmbedding(config.vocab_size, config.n_embd, config.n_layer)
        
        # Use standard Linear for SA stability
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def _make_block_mask(self, idx: torch.Tensor, sliding_window_num_blocks: int, block_size: int = 128):
        T = idx.numel()
        docs = (idx == 50256).cumsum(0)
        n_blocks = T // block_size
        docs_low = docs.reshape(n_blocks, block_size)[:, 0].contiguous()
        docs_high = docs.reshape(n_blocks, block_size)[:, -1].contiguous()
        
        def document_sliding_window_causal(b, h, q_idx, kv_idx):
             causal_mask = q_idx >= kv_idx
             document_mask = docs[q_idx] == docs[kv_idx]
             window_mask = q_idx - kv_idx < sliding_window_num_blocks * block_size
             return causal_mask & document_mask & window_mask

        sw = torch.tensor(sliding_window_num_blocks, dtype=torch.int32, device=idx.device).clamp_min(1)
        kv_idx = torch.arange(n_blocks, dtype=torch.int32, device=idx.device)
        q_idx = kv_idx[:, None]
        
        causal_bm = q_idx >= kv_idx
        window_bm = (q_idx - kv_idx) < sw
        document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
        
        dense_mask = causal_bm & window_bm & document_bm
        
        num_blocks = dense_mask.sum(dim=-1).to(torch.int32)
        indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True).to(torch.int32)
        num_blocks = num_blocks[None, None, :].contiguous()
        indices = indices[None, None, :].contiguous()
        
        return BlockMask.from_kv_blocks(num_blocks, indices, BLOCK_SIZE=block_size, mask_mod=document_sliding_window_causal)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, attn_blocksize: int = 1024, return_logits: bool = True):
        idx = idx.view(-1)
        if targets is not None:
            targets = targets.view(-1)
            
        BLOCK_SIZE = 128
        sliding_window_num_blocks = max(1, attn_blocksize // BLOCK_SIZE)
        
        block_mask = self._make_block_mask(idx, sliding_window_num_blocks, BLOCK_SIZE)
        
        x = self.transformer.wte(idx[None])
        x = apply_norm(x) 
        x0 = x
        
        ve_all = self.value_embeds(idx)
        
        skip_connections = []
        
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_all[i], x0, block_mask)
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            skip = skip_connections.pop()
            weights_skip = self.skip_weights[i] * skip
            layer_idx = self.num_encoder_layers + i
            x = self.transformer.h[layer_idx](x + weights_skip, ve_all[layer_idx], x0, block_mask)
            
        x = apply_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            
            sc = self.lm_head_softcap
            logits = sc * torch.tanh(logits / sc)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            sc = self.lm_head_softcap
            logits = sc * torch.tanh(logits / sc)
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss


# =============================================================================
# Helper Functions
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: str) -> GPT:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cfg = GPTConfig(
        vocab_size=50304,
        block_size=128,
        n_layer=12,
        n_head=6,
        n_embd=768,
        seq_len=65536
    )
    if isinstance(ckpt, dict) and "args" in ckpt:
        args = ckpt["args"]
        if isinstance(args, dict):
             get_arg = lambda k, d: args.get(k, d)
        else:
             get_arg = lambda k, d: getattr(args, k, d)
        cfg = GPTConfig(
            vocab_size=get_arg("vocab_size", 50304),
            block_size=get_arg("block_size", 128),
            n_layer=get_arg("n_layer", 12),
            n_head=get_arg("n_head", 6),
            n_embd=get_arg("n_embd", 768),
            seq_len=get_arg("sequence_length", 65536),
        )

    model = GPT(cfg, lm_head_softcap=15.0) 
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    if state_dict is None: state_dict = ckpt

    def strip_prefix(k: str) -> str:
        for p in ("_orig_mod.", "module.", "model."):
            if k.startswith(p):
                return k[len(p):]
        return k

    cleaned = {strip_prefix(k): v for k, v in state_dict.items() 
               if not k.startswith(("optimizer", "rng", "scheduler"))}
    
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as e:
        print(f"Strict load failed: {e}. Trying strict=False")
        model.load_state_dict(cleaned, strict=False)

    model.to(device)
    model.eval()
    return model


HEADER_BYTES = 256 * 4
FW_MAGIC = 20240520

def _load_data_shard(path):
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

def get_data_loader(data_path: str, num_samples: int, batch_size: int, seq_length: int, device: str):
    tokens = _load_data_shard(data_path)
    tokens = torch.from_numpy(tokens.astype(np.int64))
    num_sequences = len(tokens) // seq_length
    if num_samples == 0 or num_samples > num_sequences:
        num_samples = num_sequences
    
    data = tokens[: num_samples * seq_length].view(num_samples, seq_length)
    for i in range(0, num_samples, batch_size):
        x = data[i: i + batch_size].to(device)
        y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        yield x, y


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

RMS_EPS = 1e-6

def embed_inputs_and_mask_and_vte(model: GPT, idx: torch.Tensor, attn_blocksize: int):
    idx_flat = idx.view(-1)
    BLOCK_SIZE = 128
    sliding_window_num_blocks = max(1, attn_blocksize // BLOCK_SIZE)
    block_mask = model._make_block_mask(idx_flat, sliding_window_num_blocks, BLOCK_SIZE)
    x = model.transformer.wte(idx_flat[None]) # (1, S, C)
    x = apply_norm(x)
    ve_all = model.value_embeds(idx_flat)
    return x, x, ve_all, block_mask 

@torch.no_grad()
def run_until_layer(model: GPT, x: torch.Tensor, x0: torch.Tensor, ve_all: list, block_mask, layer_idx: int) -> torch.Tensor:
    skip_connections = []
    for i in range(len(model.transformer.h)):
        if i > layer_idx:
            break
        if i < model.num_encoder_layers:
            x = model.transformer.h[i](x, ve_all[i], x0, block_mask)
            skip_connections.append(x)
        else:
            decoder_idx = i - model.num_encoder_layers
            if not skip_connections:
                 raise ValueError("Skip connections empty in decoder phase")
            skip = skip_connections.pop()
            weights_skip = model.skip_weights[decoder_idx] * skip
            x = model.transformer.h[i](x + weights_skip, ve_all[i], x0, block_mask)
    return apply_norm(x)

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
        self.M2 += (np.dot(X.T, X) - n_new * np.outer(mean_new, mean_new))
        if self.n > 0:
            self.M2 += (self.n * n_new / total) * np.outer(delta, delta)
        self.mean += (n_new / total) * delta
        self.n = total

    def covariance(self) -> np.ndarray:
        if self.n < 2: return np.zeros((self.dim, self.dim))
        return self.M2 / (self.n - 1)

@torch.no_grad()
def compute_covariance_spectrum(model: GPT, data_loader, layer_idx: int, attn_blocksize: int) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    C = model.config.n_embd
    cov = OnlineCov(C)
    
    for idx, _ in data_loader:
        idx = idx.to(device)
        x, x0, ve_all, block_mask = embed_inputs_and_mask_and_vte(model, idx, attn_blocksize)
        h_at = run_until_layer(model, x, x0, ve_all, block_mask, layer_idx)
        if not torch.isfinite(h_at).all():
            continue
        cov.update(h_at.reshape(-1, C).float().cpu().numpy())
    
    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals

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

def get_layer_param(model: GPT, layer_idx: int, path: str) -> torch.nn.Parameter:
    layer = model.transformer.h[layer_idx]
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj

def compute_gradient_svd_spectrum(model: GPT, data_loader, layer_idx: int, param_path: str, num_samples: int, attn_blocksize: int):
    model.train()
    device = next(model.parameters()).device
    theta = get_layer_param(model, layer_idx, param_path)
    G = torch.zeros(num_samples, theta.numel(), dtype=torch.float32, device='cpu')
    
    for p in model.parameters():
        p.requires_grad_(False)
    theta.requires_grad_(True)
    
    collected = 0
    for idx, target in data_loader:
        if collected >= num_samples: break
        idx = idx.to(device)
        target = target.to(device)
        B_in = idx.size(0)
        for b in range(B_in):
            if collected >= num_samples: break
            _, loss = model(idx[b:b+1], target[b:b+1], attn_blocksize=attn_blocksize)
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            grad_flat = theta.grad.detach().reshape(-1).to('cpu')
            G[collected] = grad_flat
            collected += 1
    
    with torch.no_grad():
        s = torch.linalg.svdvals(G[:collected])
    return s.cpu().numpy()

def analyze_power_law(spectrum: np.ndarray, tail_start: int = 10, tail_finish: int = 100) -> dict:
    spectrum = np.sort(spectrum)[::-1]
    i = np.arange(1, len(spectrum) + 1)
    start_idx = min(tail_start - 1, len(spectrum) - 2)
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

def main():
    parser = argparse.ArgumentParser(description="GPT-2 FP8 LM Head SA")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing .pt checkpoints")
    parser.add_argument("--validation_data_path", type=str, required=True, help="Path to FineWeb validation .bin file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save results")
    parser.add_argument("--layers", type=str, default="all", help="Layers to analyze")
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight", help="Comma-separated parameter paths for gradient analysis")
    parser.add_argument("--task_index", type=int, default=-1, help="SLURM Array Task ID")
    parser.add_argument("--seq_length", type=int, default=65536)
    parser.add_argument("--num_samples_grad", type=int, default=512)
    parser.add_argument("--num_samples_cov", type=int, default=512)
    parser.add_argument("--cov_batch_size", type=int, default=4)
    parser.add_argument("--grad_batch_size", type=int, default=4)
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)
    parser.add_argument("--attn_blocksize", type=int, default=1024, help="Window size for flex attention")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matrices_to_analyze = [m.strip() for m in args.matrices.split(",")]
    ckpt_paths = sorted(
        list(ckpt_dir.glob("checkpoint_step*.pt")) + 
        list(ckpt_dir.glob("ckpt_step_*.pt")) + 
        list(ckpt_dir.glob("checkpoint_target*.pt")),
        key=lambda p: int(re.search(r"step_?(\d+)", p.stem).group(1))
    )
    
    if not ckpt_paths:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    if args.task_index != -1:
        if not 0 <= args.task_index < len(ckpt_paths):
            raise IndexError(f"--task_index {args.task_index} out of bounds (n={len(ckpt_paths)})")
        ckpt_paths = [ckpt_paths[args.task_index]]
        print(f"--- Array Job Mode: Processing index {args.task_index} ---")
    
    print("Loading temp model for config...")
    temp_model = load_model_from_checkpoint(str(ckpt_paths[0]), "cpu")
    layers_to_analyze = parse_layers(args.layers, temp_model.config.n_layer)
    mask_block_size = temp_model.config.block_size
    del temp_model
    gc.collect()
    torch.cuda.empty_cache()

    if args.seq_length % mask_block_size != 0:
        raise ValueError(f"--seq_length={args.seq_length} must be a multiple of block_size={mask_block_size}")
    
    print("Starting analysis...")
    print(f"Using analysis seq_length: {args.seq_length} | mask block size: {mask_block_size}")
    print(f"Using attention block size: {args.attn_blocksize}")
    print(f"Samples: cov={args.num_samples_cov} grad={args.num_samples_grad} | batch: cov={args.cov_batch_size} grad={args.grad_batch_size}")
    
    for path in tqdm(ckpt_paths):
        try:
            m = re.search(r"step_?(\d+)", path.stem)
            step_str = m.group(1) if m else "0"
            step = int(step_str)

            # Check if all files exist (OUTSIDE layer loop)
            all_exist = True
            for layer_idx in layers_to_analyze:
                cov_path = output_dir / f"cov_spectrum_{step_str}_L{layer_idx:02d}.npy"
                for matrix_name in matrices_to_analyze:
                    matrix_name_clean = matrix_name.replace('.', '_')
                    grad_path = output_dir / f"grad_spectrum_{step_str}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                    weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                    if not (cov_path.exists() and grad_path.exists() and weight_path.exists()):
                        all_exist = False
                        break
                if not all_exist:
                    break

            if all_exist:
                print(f"[info] All outputs exist for step {step_str}; skipping.")
                continue

            # Load model (OUTSIDE layer loop)
            model = load_model_from_checkpoint(str(path), device)
            checkpoint_results = []

            for layer_idx in layers_to_analyze:
                cov_path = output_dir / f"cov_spectrum_{step_str}_L{layer_idx:02d}.npy"
                if not cov_path.exists():
                    print(f"Computing covariance: {cov_path.name}")
                    dl_cov = get_data_loader(args.validation_data_path, args.num_samples_cov, args.cov_batch_size, args.seq_length, device)
                    cov_eigvals = compute_covariance_spectrum(model, dl_cov, layer_idx, args.attn_blocksize)
                    np.save(cov_path, cov_eigvals)
                else:
                    cov_eigvals = np.load(cov_path)

                cov_fit = analyze_power_law(cov_eigvals, args.tail_start, args.tail_finish)

                for matrix_name in matrices_to_analyze:
                    matrix_name_clean = matrix_name.replace('.', '_')
                    grad_path = output_dir / f"grad_spectrum_{step_str}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                    weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                    if not grad_path.exists():
                        print(f"Computing gradient SVD: {grad_path.name}")
                        dl_grad = get_data_loader(args.validation_data_path, args.num_samples_grad, args.grad_batch_size, args.seq_length, device)
                        grad_svd = compute_gradient_svd_spectrum(model, dl_grad, layer_idx, matrix_name, args.num_samples_grad, args.attn_blocksize)
                        np.save(grad_path, grad_svd)
                    else:
                        grad_svd = np.load(grad_path)

                    grad_fit = analyze_power_law(grad_svd, args.tail_start, args.tail_finish)

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
                        "step": step, "layer": layer_idx, "matrix": matrix_name,
                        "cov_alpha": float(cov_fit["alpha"]), "cov_r2": float(cov_fit["r_squared"]),
                        "grad_alpha": float(grad_fit["alpha"]), "grad_r2": float(grad_fit["r_squared"]),
                        'weight_alpha': float(weight_fit['alpha']),
                        'weight_r2': float(weight_fit['r_squared']),
                    })

            del model
            torch.cuda.empty_cache()
            
            if checkpoint_results:
                csv_path = output_dir / f"summary_step_{step_str}.csv"
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
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
