import os
import re
import math
import pathlib
import argparse
import time
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import tqdm


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


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    # Match 6_untie_embed.py (Standard Rotation: x1*cos - x2*sin)
    y1 = x1 * cos - x2 * sin
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)

        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        qh = q.transpose(1, 2).contiguous()
        kh = k.transpose(1, 2).contiguous()
        vh = v.transpose(1, 2).contiguous()

        y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss


# =============================================================================
# Helper Functions
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: str) -> GPT:
    """Load a GPT model from a checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cfg = GPTConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=6,
        n_embd=768
    )
    
    if isinstance(ckpt, dict) and "args" in ckpt:
        args = ckpt["args"]
        cfg = GPTConfig(
            vocab_size=args.get("vocab_size", 50304),
            block_size=getattr(args, "sequence_length", 1024),
            n_layer=args.get("n_layer", 12),
            n_head=args.get("n_head", 6),
            n_embd=args.get("n_embd", 768),
        )

    model = GPT(cfg)
    
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = ckpt

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
        print(f"STRICT LOAD FAILED: {e}")
        # Optionally, for now, we can still fall back but failing loudly is better
        # model.load_state_dict(cleaned, strict=False)
        raise e

    del ckpt, state_dict, cleaned
    gc.collect()

    model.to(device)
    model.eval()
    return model


HEADER_BYTES = 256 * 4
FW_MAGIC = 20240520


def _load_data_shard(path):
    """Load a FineWeb data shard."""
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
    """Create a simple data loader from a token file."""
    tokens = _load_data_shard(data_path)
    stride = seq_length + 1
    num_sequences = len(tokens) // stride

    if num_samples == 0 or num_samples > num_sequences:
        num_samples = num_sequences
    
    # Slice raw data into (N, T+1) chunks
    data = tokens[: num_samples * stride].astype(np.int64)
    data = torch.from_numpy(data).view(num_samples, stride)
    
    for i in range(0, num_samples, batch_size):
        batch = data[i: i + batch_size].to(device)
        x = batch[:, :-1].contiguous() # 0..T-1
        y = batch[:, 1:].contiguous()  # 1..T
        yield x, y


def parse_layers(layers_str: str, total_layers: int) -> list[int]:
    """Parse layer specification string."""
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
    valid_layers = sorted([l for l in out if 0 <= l < total_layers])
    return valid_layers


RMS_EPS = 1e-6


def embed_inputs(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """Get initial embeddings, including the pre-block RMSNorm used in 6_untie_embed.py.

    The training script applies F.rms_norm to the raw token embeddings BEFORE
    passing them into the first transformer block:
        x = wte(idx)
        x = F.rms_norm(x, (x.size(-1),))   # <-- this must be replicated here
        for block in h: x = block(x)
    """
    x = model.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    return x


@torch.no_grad()
def run_until_layer(model: GPT, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run forward pass up to and including the specified layer, return normalized output."""
    for i in range(layer_idx + 1):
        x = model.transformer.h[i](x)
    return F.rms_norm(x, (x.size(-1),), eps=RMS_EPS)


def get_layer_param(model: GPT, layer_idx: int, path: str) -> torch.nn.Parameter:
    """Get a parameter from a specific layer by path."""
    layer = model.transformer.h[layer_idx]
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_global_param(model: GPT, path: str) -> torch.nn.Parameter:
    """Get a global parameter (not layer-specific) by path."""
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class OnlineCov:
    """Numerically stable online covariance computation."""
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
def compute_covariance_spectrum(model: GPT, data_loader, layer_idx: int) -> np.ndarray:
    """Compute the eigenvalue spectrum of activation covariance at a given layer."""
    model.eval()
    device = next(model.parameters()).device
    C = model.config.n_embd
    cov = OnlineCov(C)
    
    for idx, _ in data_loader:
        idx = idx.to(device)
        x = embed_inputs(model, idx)
        h_at = run_until_layer(model, x, layer_idx)
        if not torch.isfinite(h_at).all():
            continue
        cov.update(h_at.reshape(-1, C).float().cpu().numpy())
    
    eigvals = np.linalg.eigvalsh(cov.covariance())[::-1]
    return eigvals


def compute_gradient_svd_spectrum(model: GPT, data_loader, layer_idx: int, param_path: str, num_samples: int):
    """
    Compute SVD spectrum of per-sample gradients for a given parameter.
    
    Args:
        model: The GPT model
        data_loader: Iterator yielding (input, target) batches
        layer_idx: Layer index (-1 for global parameters like wte)
        param_path: Dot-separated path to parameter (e.g., "attn.c_proj.weight")
        num_samples: Number of gradient samples to collect
    
    Returns:
        Singular values of the gradient matrix
    """
    model.train()
    device = next(model.parameters()).device
    
    if layer_idx >= 0:
        theta = get_layer_param(model, layer_idx, param_path)
    else:
        theta = get_global_param(model, param_path)
    
    P = theta.numel()
    G = torch.zeros(num_samples, P, dtype=torch.float32, device='cpu')
    
    for p in model.parameters():
        p.requires_grad_(False)
    theta.requires_grad_(True)
    
    collected = 0
    for idx, target in data_loader:
        if collected >= num_samples: 
            break
        idx = idx.to(device)
        target = target.to(device)
        B = idx.size(0)
        
        for b in range(B):
            if collected >= num_samples: 
                break
            _, loss_b = model(idx[b:b+1], target[b:b+1])
            model.zero_grad(set_to_none=True)
            loss_b.backward(retain_graph=False)
            grad_flat = theta.grad.detach().reshape(-1).to('cpu')
            G[collected, :grad_flat.numel()] = grad_flat
            collected += 1
    
    with torch.no_grad():
        s = torch.linalg.svdvals(G[:collected])
    return s.cpu().numpy()


def analyze_power_law(spectrum: np.ndarray, tail_start: int = 10, tail_finish: int = 100) -> dict:
    """Fit a power law to the spectrum tail."""
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
    parser = argparse.ArgumentParser(description="GPT-2 Untied Embeddings Spectrum Analysis")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Directory containing .pt checkpoints")
    parser.add_argument("--validation_data_path", type=str, required=True, 
                        help="Path to FineWeb validation .bin file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Where to save results")
    parser.add_argument("--layers", type=str, default="all", 
                        help="Layers to analyze (e.g., '0,5,11' or '0-5' or 'all')")
    parser.add_argument("--matrices", type=str, default="attn.c_proj.weight",
                        help="Comma-separated parameter paths for gradient analysis")
    parser.add_argument("--matrix_name", type=str, default=None,
                        help="Deprecated alias for --matrices")
    
    # SLURM array job support
    parser.add_argument("--task_index", type=int, default=-1, 
                        help="SLURM Array Task ID. If -1, runs all sequentially.")
    
    # Sample counts
    parser.add_argument("--num_samples_grad", type=int, default=512,
                        help="Number of gradient samples to collect")
    parser.add_argument("--num_samples_cov", type=int, default=32768,
                        help="Number of samples for covariance estimation")
    parser.add_argument("--cov_batch_size", type=int, default=128)
    parser.add_argument("--grad_batch_size", type=int, default=1)
    
    # Power law fitting
    parser.add_argument("--tail_start", type=int, default=30)
    parser.add_argument("--tail_finish", type=int, default=150)
    
    args = parser.parse_args()
    if args.matrix_name is not None:
        if args.matrices != parser.get_default("matrices") and args.matrices != args.matrix_name:
            print("[warn] Both --matrices and deprecated --matrix_name were provided; using --matrices.")
        else:
            print("[warn] --matrix_name is deprecated; use --matrices instead.")
            args.matrices = args.matrix_name

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
    n_layer = temp_model.config.n_layer
    block_size = temp_model.config.block_size
    
    del temp_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Config: n_layer={n_layer}, analyzing layers: {layers_to_analyze}")
    print("Starting analysis...")

    for path in tqdm(ckpt_paths, desc="Processing checkpoints"):
        try:
            m = re.search(r"step_?(\d+)", path.stem)
            step_str = m.group(1) if m else "0"
            step = int(step_str)

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
                
                if all_files_exist:
                    print(f"All files for Step {step} exist. Skipping.")
                    continue
    
                model = load_model_from_checkpoint(str(path), device)
                checkpoint_results = []
    
                for layer_idx in layers_to_analyze:
                    cov_path = output_dir / f"cov_spectrum_{step_str}_L{layer_idx:02d}.npy"
                    if cov_path.exists():
                        print(f"Skipping existing: {cov_path.name}")
                        cov_eigvals = np.load(cov_path)
                    else:
                        print(f"Computing covariance: {cov_path.name}")
                        dl_cov = get_data_loader(args.validation_data_path, args.num_samples_cov, 
                                                 args.cov_batch_size, block_size, device)
                        cov_eigvals = compute_covariance_spectrum(model, dl_cov, layer_idx)
                        np.save(cov_path, cov_eigvals)
                    
                    cov_fit = analyze_power_law(cov_eigvals, args.tail_start, args.tail_finish)
    
                    grad_path = output_dir / f"grad_spectrum_{step_str}_L{layer_idx:02d}_{matrix_name_clean}.npy"
                    weight_path = grad_path.with_name(grad_path.name.replace("grad_spectrum_", "weight_spectrum_"))
                    if grad_path.exists():
                        print(f"Skipping existing: {grad_path.name}")
                        grad_svd = np.load(grad_path)
                    else:
                        print(f"Computing gradient SVD: {grad_path.name}")
                        dl_grad = get_data_loader(args.validation_data_path, args.num_samples_grad, 
                                                  args.grad_batch_size, block_size, device)
                        grad_svd = compute_gradient_svd_spectrum(model, dl_grad, layer_idx, 
                                                                 matrix_name, args.num_samples_grad)
                        np.save(grad_path, grad_svd)
    
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
                        "step": step,
                        "layer": layer_idx,
                        "matrix": matrix_name,
                        "cov_alpha": float(cov_fit["alpha"]),
                        "cov_r2": float(cov_fit["r_squared"]),
                        "grad_alpha": float(grad_fit["alpha"]),
                        "grad_r2": float(grad_fit["r_squared"]),
                        'weight_alpha': float(weight_fit['alpha']),
                        'weight_r2': float(weight_fit['r_squared']),
                    })
                    # end of matrix loop

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

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
                        print(f"Appended results to {csv_path}")
                    except Exception as e:
                        print(f"Error appending CSV: {e}")
                        new_df.to_csv(output_dir / f"summary_step_{step_str}_new.csv", index=False)
                else:
                    new_df.to_csv(csv_path, index=False)
                    print(f"Created {csv_path}")

        except Exception as e:
            print(f"Error processing {path}: {e}")
            import traceback
            traceback.print_exc()

    print("Done!")


if __name__ == "__main__":
    main()
