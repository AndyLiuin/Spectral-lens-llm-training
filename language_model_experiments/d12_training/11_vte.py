# =================================================================
# GPT-2 Training with VTE (Value Token Embedding) Architecture
# (vte embeddings + skip connections + lambda parameters + flex attention)
# =================================================================

# Standard library imports
import os
import sys
import math
import glob
import struct
import inspect
import time
import argparse
import csv
import json
import io
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple, Literal, Optional

# Third-party imports
import numpy as np
import tiktoken
import wandb

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

# Flex attention imports
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# =================================================================
# Argument Parser
# =================================================================

def get_args():
    parser = argparse.ArgumentParser(description="GPT-2 Training with VTE Architecture")
    
    # Data paths
    parser.add_argument("--train_pattern", type=str, required=True, help="Glob pattern for training data shards")
    parser.add_argument("--val_pattern", type=str, default="", help="Glob pattern for validation data shards")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs")
    
    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    
    # Batching
    parser.add_argument("--batch_size", type=int, default=8, help="Global batch size (sequences)")
    parser.add_argument("--device_batch_size", type=int, default=1, help="Sequences per GPU (B=1 for flex attention)")
    parser.add_argument("--sequence_length", type=int, default=65536, help="Tokens per sequence")
    
    # Optimizer learning rates (4-optimizer setup)
    parser.add_argument("--embed_lr", type=float, default=0.6, help="Adam LR for embeddings (wte + vte)")
    parser.add_argument("--head_lr", type=float, default=0.008, help="Adam LR for output head")
    parser.add_argument("--muon_lr", type=float, default=0.04, help="Muon LR for matrix parameters")
    parser.add_argument("--scalar_lr", type=float, default=0.04, help="Adam LR for scalar parameters")
    
    # Muon momentum warmup
    parser.add_argument("--muon_momentum_init", type=float, default=0.85, help="Initial Muon momentum")
    parser.add_argument("--muon_momentum_final", type=float, default=0.95, help="Final Muon momentum")
    parser.add_argument("--muon_momentum_warmup_steps", type=int, default=500, help="Steps to warmup momentum")
    
    # LR schedule
    parser.add_argument("--warmup_frac", type=float, default=0.0, help="Fraction of steps for warmup")
    parser.add_argument("--warmdown_frac", type=float, default=0.1, help="Fraction of steps for warmdown")
    
    # Training
    parser.add_argument("--num_iterations", type=int, default=20000, help="Maximum training iterations")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                        help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    parser.add_argument("--loss_threshold", type=float, default=3.3,
                        help="[const_loss mode] Early stop when val loss < threshold")
    parser.add_argument("--stop_epoch_frac", type=float, default=None,
                        help="[epoch mode] Stop at this epoch fraction (e.g., 0.5). Required if stop_mode='epoch'.")
    
    # Flexible window attention warmup
    parser.add_argument("--window_min", type=int, default=64, help="Minimum window size (start of warmup)")
    parser.add_argument("--window_max", type=int, default=1792, help="Maximum window size (end of warmup)")
    parser.add_argument("--window_warmup_steps", type=int, default=4000, help="Steps to warmup window from min to max")
    
    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    parser.add_argument("--val_every_steps", type=int, default=100,
                        help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    parser.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                        help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="Tokens for validation")

    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    parser.add_argument("--checkpoint_every_steps", type=int, default=400,
                        help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    parser.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                        help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")
    
    # Numerics
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--tensorcores", action="store_true", help="Enable tensor cores")
    parser.add_argument("--use_cudnn_attn", action="store_true", help="Use CUDNN attention backend")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="gpt2-dynamics", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team")
    parser.add_argument("--wandb_run_name", type=str, default="10_vte_gpt2", help="WandB run name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1, help="Log to wandb every N steps")
    
    return parser.parse_args()

# =================================================================
# Initial Setup
# =================================================================

print(f"Running pytorch {torch.version.__version__}")

def print0(*print_args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*print_args, **kwargs)
    
# GPU inventory function
def print_gpu_inventory():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    n = torch.cuda.device_count()
    names = []
    total_mem_gb = 0.0
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        mem_gb = props.total_memory / (1024**3)
        cc = f"{props.major}.{props.minor}"
        print(f"GPU {i}: {name} | {mem_gb:.1f} GB | Compute Capability {cc}")
        names.append(name)
        total_mem_gb += mem_gb
    h200_count = sum("H200" in nm.upper() for nm in names)
    print(f"\nDetected {n} CUDA device(s). H200 count guess: {h200_count}. Total VRAM: {total_mem_gb:.1f} GB.")

print_gpu_inventory()

# =================================================================
# Helper Functions
# =================================================================

def get_peak_mem_mib():
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() // 1024 // 1024)
    return 0

def get_reserved_mem_mib():
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_reserved() // 1024 // 1024)
    return 0

def get_free_total_mib():
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return int(free // 1024 // 1024), int(total // 1024 // 1024)
        except Exception:
            pass
    return 0, 0

def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# =================================================================
# Optimizer Components
# =================================================================

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                gg = g
                if group['nesterov']:
                    gg = gg.add(buf, alpha=momentum)

                gg = zeropower_backend(gg, steps=group['backend_steps'])
                gg *= max(1, gg.size(0) / gg.size(1)) ** 0.5

                p.data.add_(gg.to(dtype=p.data.dtype), alpha=-lr)

# =================================================================
# Model Components
# =================================================================

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
        assert B == 1, "Must use batchsize = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        v = ((1 - self.lamb) * v + self.lamb * vi.view_as(v)).to(v.dtype)
        cos, sin = self.rotary(q)
        q, k = apply_norm(q), apply_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flex_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
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
        x = x + self.attn(apply_norm(x), vi, block_mask)
        x = x + self.mlp(apply_norm(x))
        return x

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
        self.config = config

    def forward(self, idx, target, attn_blocksize):
        idx = idx.view(-1)
        target = target.view(-1)

        docs = (idx == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask
    
        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
        
        x = self.transformer.wte(idx[None])
        x = apply_norm(x)
        x0 = x
        vi = self.transformer.vte(idx[None]).chunk(self.config.n_layer, dim=-1)

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, vi[i], x0, block_mask)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, vi[self.num_encoder_layers+i], x0, block_mask)
        
        x = apply_norm(x) 
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            
        return loss

# =================================================================
# Data Loading
# =================================================================

FW_MAGIC = 20240520
HEADER_BYTES = 256 * 4
_RAW_DTYPE_CACHE = {}

def _detect_format(path):
    with open(path, "rb") as f:
        head = f.read(10)
    if head.startswith(b"\x93NUMPY"):
        return "npy", None
    if os.path.getsize(path) >= HEADER_BYTES:
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
        if header.size >= 3 and header[0] == FW_MAGIC:
            return "fineweb", int(header[1])
    return "raw", None

def _inspect_raw_dtype(path, sample_bytes=4*1024*1024):
    if path in _RAW_DTYPE_CACHE:
        return _RAW_DTYPE_CACHE[path]
    size = os.path.getsize(path)
    cand = []
    if size % 2 == 0:
        cand.append(np.uint16)
    if size % 4 == 0:
        cand.append(np.uint32)
    if not cand:
        raise AssertionError(f"{path}: size {size} is not divisible by 2 or 4")
    with open(path, "rb") as f:
        buf = f.read(min(sample_bytes, size))
    choice = None
    if np.uint16 in cand:
        sample = np.frombuffer(buf[: (len(buf)//2)*2], dtype=np.uint16)
        if sample.size > 0 and sample.max() < 65536:
            choice = np.uint16
    if choice is None and np.uint32 in cand:
        choice = np.uint32
    _RAW_DTYPE_CACHE[path] = choice
    return choice

def _peek_data_shard(path):
    fmt, ver = _detect_format(path)
    if fmt == "fineweb":
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
        return int(header[2])
    elif fmt == "npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 1:
            raise AssertionError(f"{path}: npy shard must be 1D, got {arr.shape}")
        return int(arr.size)
    else:
        dtype = _inspect_raw_dtype(path)
        itemsize = np.dtype(dtype).itemsize
        return os.path.getsize(path) // itemsize

def _load_data_shard(path):
    fmt, ver = _detect_format(path)
    if fmt == "fineweb":
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
            ntok = int(header[2])
            cur = f.tell()
            f.seek(0, os.SEEK_END); total = f.tell(); f.seek(cur)
            data_bytes = total - cur
            if data_bytes == ntok * 2:
                dtype = np.uint16
            elif data_bytes == ntok * 4:
                dtype = np.uint32
            else:
                raise AssertionError(f"{path}: cannot infer dtype from sizes")
            tokens = np.frombuffer(f.read(), dtype=dtype)
    elif fmt == "npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        tokens = np.asarray(arr)
    else:
        dtype = _inspect_raw_dtype(path)
        tokens = np.memmap(path, mode="r", dtype=dtype)

    if tokens.dtype != np.uint16:
        sample = np.asarray(tokens[:1_000_000])
        if sample.size == 0 or sample.max() < 65536:
            tokens = np.asarray(tokens, dtype=np.uint16)
        else:
            raise AssertionError(f"{path}: token ids exceed 65535")
    return np.asarray(tokens)

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            ntok_total += shard_ntok
        self.ntok_total = ntok_total

        self.current_shard = None
        self.tokens = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T, W = self.B, self.T, self.num_processes
        global_bt = W * B * T

        if self.current_position + B * T + 1 >= len(self.tokens):
            self.advance()

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += global_bt
        if self.current_position + global_bt + 1 >= len(self.tokens):
            self.advance()
        return x, y

# =================================================================
# Checkpoint Management
# =================================================================

def save_training_checkpoint(step, raw_model, optimizers, out_dir, val_loss=None):
    os.makedirs(out_dir, exist_ok=True)
    obj = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
        "optimizers": [opt.state_dict() for opt in optimizers],
        "val_loss": val_loss,
    }
    fname = f"checkpoint_step{step:06d}.pt"
    tmp_path = os.path.join(out_dir, fname + ".tmp")
    final_path = os.path.join(out_dir, fname)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path

# =================================================================
# Main Training Function
# =================================================================

def main():
    args = get_args()
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        device_type = 'cuda'
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_type = 'cuda' if device == "cuda" else "cpu"

    set_seed(args.seed, ddp_rank)

    # Logging setup
    logfile = None
    csv_file = None
    csv_writer = None

    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "train.log")

        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        csv_path = os.path.join(args.output_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "timestamp", "kind", "step", "train_loss", "val_loss",
            "lr_embed", "lr_head", "lr_muon", "lr_scalar", "muon_momentum",
            "grad_norm", "step_time_ms", "tokens_per_sec",
            "train_tokens", "val_tokens", "total_tokens",
            "train_time_s", "wall_time_s", "peak_mem_mib"
        ])
        csv_writer.writeheader()

        with open(logfile, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"VTE Training started at {now_ts()}\n")
            f.write(f"{'='*60}\n\n")

    def log_metrics(row):
        if csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()

    def log_print(*log_args, **kwargs):
        if master_process:
            msg = " ".join(str(a) for a in log_args)
            print(msg, **kwargs)
            if logfile:
                with open(logfile, "a") as f:
                    f.write(msg + "\n")

    # Calculate schedule
    total_steps = args.num_iterations
    warmup_steps = int(round(args.warmup_frac * total_steps))
    warmdown_steps = int(round(args.warmdown_frac * total_steps))
    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)

    # Batching
    B, T = args.device_batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.batch_size % (B * ddp_world_size) == 0
    grad_accum_steps = args.batch_size // (B * ddp_world_size)
    total_batch_tokens = tokens_per_fwdbwd * grad_accum_steps

    # Print configuration
    log_print(f"\n{'='*60}")
    log_print("CONFIGURATION: VTE ARCHITECTURE (10_vte)")
    log_print(f"{'='*60}")
    log_print(f"Device: {device} | DDP: {ddp} | World Size: {ddp_world_size}")
    log_print(f"\nBatching (Flex Attention - Single Long Sequences):")
    log_print(f"  Device batch size: {B} sequence")
    log_print(f"  Sequence length: {T:,} tokens")
    log_print(f"  Total batch size: {args.batch_size} sequences")
    log_print(f"  Tokens per forward/backward: {tokens_per_fwdbwd:,}")
    log_print(f"  Gradient accumulation steps: {grad_accum_steps}")
    log_print(f"  Total tokens per step: {total_batch_tokens:,}")

    log_print(f"\nOptimization (4 Optimizers):")
    log_print(f"  1. Adam (wte+vte): lr={args.embed_lr}")
    log_print(f"  2. Adam (lm_head): lr={args.head_lr}")
    log_print(f"  3. Muon (matrix): lr={args.muon_lr}, momentum={args.muon_momentum_init}→{args.muon_momentum_final}")
    log_print(f"  4. Adam (scalar): lr={args.scalar_lr}")

    log_print(f"\nLR Schedule:")
    log_print(f"  Total steps: {total_steps:,}")
    log_print(f"  Warmup: {warmup_steps:,} steps")
    log_print(f"  Plateau: {plateau_steps:,} steps")
    log_print(f"  Warmdown: {warmdown_steps:,} steps")
    log_print(f"{'='*60}\n")

    # Precision
    def _sync_device():
        if device_type == 'cuda':
            torch.cuda.synchronize()

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # Model
    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"Model parameters: {total_params:,}")
    log_print(f"  Architecture: VTE + UNet + Fixed Window Flex Attention")

    model.train().to(device)

    # CUDNN Attention Configuration
    if args.use_cudnn_attn:
        try:
            from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
            enable_cudnn_sdp(True)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            log_print("Enabled CUDNN attention")
        except Exception as e:
            log_print(f"Warning: Could not enable CUDNN attention: {e}")

    # Compile
    if args.compile:
        log_print("Compiling model with torch.compile()...")
        if inductor_config:
            inductor_config.coordinate_descent_tuning = True
        model = torch.compile(model)

    # Data
    train_loader = DistributedDataLoader(args.train_pattern, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, ddp_rank, ddp_world_size) if args.val_pattern else None

    log_print(f"\nData:")
    log_print(f"  Train shards: {len(train_loader.files)} | Total tokens: {train_loader.ntok_total:,}")
    if val_loader:
        log_print(f"  Val shards: {len(val_loader.files)} | Total tokens: {val_loader.ntok_total:,}")

    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_max_steps = args.val_tokens // (B * T * ddp_world_size)
    log_print(f"  Val steps per eval: {val_max_steps}")

    # DDP wrap
    # Validation cadence: step-based takes precedence over epoch-fraction-based
    if args.val_every_steps > 0:
        val_loss_every = args.val_every_steps
    elif args.val_every_epoch_frac > 0:
        val_loss_every = max(1, int(round(args.val_every_epoch_frac * epoch_steps)))
    else:
        val_loss_every = 0

    # Checkpoint cadence: step-based takes precedence over epoch-fraction-based
    if args.checkpoint_every_steps > 0:
        checkpoint_every = args.checkpoint_every_steps
    elif args.save_every_epoch_frac > 0:
        checkpoint_every = max(1, int(round(args.save_every_epoch_frac * epoch_steps)))
    else:
        checkpoint_every = 0


    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Optimizers
    log_print(f"\nConfiguring 4-optimizer setup...")
    
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2]
    
    skip_w = getattr(raw_model, "skip_weights", None)
    if skip_w is not None:
        scalar_params.append(skip_w)

    optimizer1 = torch.optim.Adam(
        [raw_model.transformer.wte.weight, raw_model.transformer.vte.weight],
        lr=args.embed_lr, betas=(0.9, 0.95), fused=(device_type=='cuda')
    )
    optimizer2 = torch.optim.Adam(
        [raw_model.lm_head.weight],
        lr=args.head_lr, betas=(0.9, 0.95), fused=(device_type=='cuda')
    )
    optimizer3 = Muon(matrix_params, lr=args.muon_lr, momentum=args.muon_momentum_init)
    optimizer4 = torch.optim.Adam(
        scalar_params, lr=args.scalar_lr, betas=(0.9, 0.95), fused=(device_type=='cuda')
    )
    
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    # LR Scheduler
    def get_lr(it):
        it = min(it, total_steps)
        if warmup_steps > 0 and it < warmup_steps:
            return (it + 1) / warmup_steps
        decay_start = warmup_steps + plateau_steps
        if warmdown_steps <= 0 or it < decay_start:
            return 1.0
        decay_ratio = (total_steps - it) / max(1, warmdown_steps)
        return max(0.0, decay_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # WandB
    wandb_run = None
    if master_process and args.wandb_mode != "disabled":
        os.makedirs(args.output_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", args.output_dir)

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=["vte", "flex_attention", "unet"],
            mode=args.wandb_mode,
            config=vars(args),
        )

        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.define_metric("loss/*", step_metric="global_step")
        wandb.define_metric("lr/*", step_metric="global_step")
        wandb.define_metric("perf/*", step_metric="global_step")
        wandb.define_metric("mem/*", step_metric="global_step")

    # Save step-0 checkpoint
    if master_process:
        ckpt0_path = save_training_checkpoint(0, raw_model, optimizers, args.output_dir, val_loss=None)
        log_print(f"   💾 Saved initial checkpoint (step 0): {ckpt0_path}")

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Training loop
    training_time_s = 0.0
    train_tokens = 0
    val_tokens_count = 0
    counted_steps = 0
    skip_warmup_steps = 10
    wall_t0 = time.time()

    step = 0
    stop_now = False
    saved_final_ckpt = False
    val_loss = float("nan")

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    free_mib, total_mib = get_free_total_mib()
    log_print(f"\nGPU Memory: {get_peak_mem_mib()} MiB allocated, {free_mib}/{total_mib} MiB free/total")
    log_print(f"\n{'='*60}")
    log_print("TRAINING STARTED")
    log_print(f"{'='*60}\n")

    while True:
        last_step = (step == args.num_iterations)

        # Window warmup: linearly increase from window_min to window_max over window_warmup_steps
        warmup_frac = min(step / args.window_warmup_steps, 1.0)
        window_size = args.window_min + warmup_frac * (args.window_max - args.window_min)
        attn_blocksize = torch.tensor(64 * (window_size // 64), dtype=torch.int, device='cuda')

        # Compute muon momentum for this step (needed for both validation logging and optimizer update)
        frac = min(step / args.muon_momentum_warmup_steps, 1.0)
        muon_momentum_current = (1 - frac) * args.muon_momentum_init + frac * args.muon_momentum_final

        run_validation = (val_loader is not None) and (val_loss_every > 0) and (step % val_loss_every == 0 or last_step)

        if run_validation:
            _sync_device()
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(val_max_steps):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
                    with ctx:
                        loss_v = model(xv, yv, attn_blocksize)
                    val_loss_t += loss_v.detach()
                val_loss_t /= max(1, val_max_steps)
                if ddp:
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                val_loss = val_loss_t.item()

            val_batch_tokens = val_max_steps * B * T * ddp_world_size
            val_tokens_count += val_batch_tokens
            total_tokens = train_tokens + val_tokens_count

            # Compute epoch info
            tokens_per_epoch = train_loader.ntok_total
            current_epoch = train_tokens / tokens_per_epoch if tokens_per_epoch > 0 else 0
            epoch_pct = (current_epoch % 1) * 100
            
            # Compute throughput
            avg_throughput = train_tokens / training_time_s if training_time_s > 0 else 0
            
            # Get memory stats
            peak_mem = get_peak_mem_mib()
            reserved_mem = get_reserved_mem_mib()
            free_mem, total_mem = get_free_total_mib()
            
            # Comprehensive validation printout
            log_print(f"\n{'─'*60}")
            log_print(f"[VAL] Step {step:,}/{args.num_iterations:,} | Epoch {int(current_epoch)+1} ({epoch_pct:.1f}%)")
            log_print(f"      Window: {int(attn_blocksize)} tokens (scheduled from {args.window_min}→{args.window_max})")
            log_print(f"      Val Loss: {val_loss:.6f} | Target: {args.loss_threshold}")
            log_print(f"      LR: embed={optimizers[0].param_groups[0]['lr']:.4g}, head={optimizers[1].param_groups[0]['lr']:.4g}, muon={optimizers[2].param_groups[0]['lr']:.4g}, scalar={optimizers[3].param_groups[0]['lr']:.4g}")
            log_print(f"      Muon momentum: {muon_momentum_current:.4f}")
            log_print(f"      Train Time: {training_time_s:.1f}s | Wall: {time.time() - wall_t0:.1f}s")
            log_print(f"      Train Tokens: {train_tokens:,} ({train_tokens/1e9:.2f}B)")
            log_print(f"      Val Tokens: {val_tokens_count:,} ({val_tokens_count/1e9:.2f}B)")
            log_print(f"      Total Tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
            log_print(f"      Throughput: {avg_throughput:,.0f} tok/s")
            log_print(f"      Memory: {peak_mem} MiB peak | {reserved_mem} MiB reserved | {free_mem} MiB free")
            log_print(f"{'─'*60}\n")

            if master_process and wandb_run is not None:
                wandb.log({
                    "global_step": step,
                    "loss/val": val_loss,
                    "val/loss": val_loss,
                    "val/epoch": current_epoch,
                    "val/window_size": int(attn_blocksize),
                })

            # Stopping criteria - mutually exclusive modes
            if args.stop_mode == "const_loss":
                # Mode 1: Stop by constant loss threshold
                if val_loss < args.loss_threshold and step > 0:
                    log_print(f"🎯 [EARLY STOP] Val loss {val_loss:.4f} < threshold {args.loss_threshold:.4f}")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        log_print(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True
                    stop_now = True
            elif args.stop_mode == "epoch":
                # Mode 2: Stop by epoch fraction
                tokens_per_epoch = train_loader.ntok_total
                epoch_frac_current = train_tokens / tokens_per_epoch if tokens_per_epoch > 0 else 0.0
                if (args.stop_epoch_frac is not None) and (epoch_frac_current >= args.stop_epoch_frac):
                    log_print(f"🎯 [EPOCH STOP] Reached {epoch_frac_current:.3f} epochs (target: {args.stop_epoch_frac})")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        log_print(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True
                    stop_now = True

            if last_step and master_process and not saved_final_ckpt:
                ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                log_print(f"   Final checkpoint saved: {ckpt_path}")
                saved_final_ckpt = True

            model.train()

        if last_step or stop_now:
            break

        # Training step
        model.train()
        _sync_device()
        t0 = time.perf_counter()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        step_losses = []
        for gas in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (gas == grad_accum_steps - 1)
            
            with ctx:
                loss = model(x, y, attn_blocksize)
                (loss / grad_accum_steps).backward()
                step_losses.append(loss.detach())

            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            norm_val = float(grad_norm) if grad_norm is not None else -1.0
        else:
            norm_val = -1.0

        # Set momentum (already computed earlier in the loop)
        optimizers[2].param_groups[0]['momentum'] = muon_momentum_current

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        _sync_device()
        step_dt = time.perf_counter() - t0

        train_tokens += total_batch_tokens

        if step >= skip_warmup_steps:
            training_time_s += step_dt
            counted_steps += 1

        if step_losses:
            step_loss_t = torch.stack(step_losses).mean()
        else:
            step_loss_t = torch.tensor(0.0, device=device)

        if ddp:
            dist.all_reduce(step_loss_t, op=dist.ReduceOp.AVG)
        train_loss_scalar = step_loss_t.item()

        # Logging
        if master_process:
            lr_embed = optimizers[0].param_groups[0]["lr"]
            lr_muon = optimizers[2].param_groups[0]["lr"]
            toks_now = total_batch_tokens / max(step_dt, 1e-9)

            total_tokens_now = train_tokens + val_tokens_count
            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"win {int(attn_blocksize):4d} │ "
                f"loss {train_loss_scalar:.4f} │ "
                f"gnorm {norm_val:6.2f} │ "
                f"lr {lr_embed:.3g}/{lr_muon:.3g} │ "
                f"μ {muon_momentum_current:.3f} │ "
                f"{step_dt*1000:6.1f}ms │ "
                f"{toks_now/1000:5.1f}k tok/s │ "
                f"tokens {total_tokens_now/1e6:.1f}M"
            )
            if device_type == "cuda":
                log_str += f" │ mem {get_peak_mem_mib()}MiB"
            log_print(log_str)

            log_metrics({
                "timestamp": now_ts(),
                "kind": "train", "step": step + 1,
                "train_loss": f"{train_loss_scalar:.6f}", "val_loss": "",
                "lr_embed": f"{lr_embed:.6g}", "lr_head": f"{optimizers[1].param_groups[0]['lr']:.6g}",
                "lr_muon": f"{lr_muon:.6g}", "lr_scalar": f"{optimizers[3].param_groups[0]['lr']:.6g}",
                "muon_momentum": f"{muon_momentum_current:.4f}",
                "grad_norm": f"{norm_val:.4f}" if norm_val >= 0 else "",
                "step_time_ms": f"{step_dt*1000:.1f}", "tokens_per_sec": f"{toks_now:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens_count,
                "total_tokens": train_tokens + val_tokens_count,
                "train_time_s": f"{training_time_s:.1f}",
                "wall_time_s": f"{time.time() - wall_t0:.1f}",
                "peak_mem_mib": get_peak_mem_mib()
            })

            if wandb_run is not None and ((step + 1) % args.wandb_log_every == 0):
                wandb.log({
                    "global_step": step + 1,
                    "loss/train": train_loss_scalar,
                    "train/loss": train_loss_scalar,
                    "train/grad_norm": norm_val if norm_val >= 0 else None,
                    "lr/embed": lr_embed,
                    "lr/muon": lr_muon,
                    "perf/step_time_ms": step_dt * 1000.0,
                    "perf/tokens_per_sec": toks_now,
                    "mem/peak_mib": get_peak_mem_mib(),
                })

        # Periodic checkpoint
        if master_process and checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
            ckpt_path = save_training_checkpoint(step + 1, raw_model, optimizers, args.output_dir, val_loss)
            log_print(f"   💾 Checkpoint saved: {ckpt_path}")

        step += 1

    # Training complete
    log_print(f"\n{'='*60}")
    log_print("TRAINING COMPLETE")
    log_print(f"{'='*60}")

    wall_time = time.time() - wall_t0
    toks_per_sec = (train_tokens / training_time_s) if training_time_s > 0 else 0

    log_print(f"\nFinal Statistics:")
    log_print(f"  Steps completed: {step:,}")
    log_print(f"  Training tokens: {train_tokens:,}")
    log_print(f"  Training time: {training_time_s:.1f}s")
    log_print(f"  Wall time: {wall_time:.1f}s")
    log_print(f"  Average throughput: {toks_per_sec:,.0f} tok/s")
    log_print(f"  Final val loss: {val_loss:.6f}")

    if device_type == "cuda":
        log_print(f"  Peak memory: {get_peak_mem_mib()} MiB")

    if csv_file:
        csv_file.close()
    if ddp:
        destroy_process_group()

    if master_process and wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    log_print(f"\nOutput directory: {args.output_dir}")
    log_print(f"{'='*60}\n")

if __name__ == "__main__":
    main()