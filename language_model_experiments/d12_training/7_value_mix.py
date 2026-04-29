# =================================================================
# GPT-2 Training with Value Mix Architecture
# (v1 mixing + skip connections + lambda parameters)
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

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

# =================================================================
# Initial Setup
# =================================================================

print(f"Running pytorch {torch.version.__version__}")

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)
    
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
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=3).type_as(x)

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
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

    def forward(self, x, v1=None):
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
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        
        return y, v1

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

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

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
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = F.rms_norm(x, (x.size(-1),))
        
        if targets is not None:
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()
            loss = None
            
        if not return_logits:
            logits = None
            
        return logits, loss

FW_MAGIC = 20240520
HEADER_BYTES = 256 * 4
_RAW_DTYPE_CACHE: dict[str, np.dtype] = {}


def read_self_code() -> str:
    try:
        with open(sys.argv[0], "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<could not read source file>"


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_master(rank: int) -> bool:
    return rank == 0


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def nvidia_smi() -> str:
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return r.stdout
    except Exception as e:
        return f"<nvidia-smi failed: {e}>"


# =============================================================================
# Robust shard reader (FineWeb header / raw / .npy)
# =============================================================================

def _detect_format(path: str) -> Tuple[Literal["fineweb", "raw", "npy"], Optional[int]]:
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


def _inspect_raw_dtype(path: str, sample_bytes: int = 4 * 1024 * 1024) -> np.dtype:
    if path in _RAW_DTYPE_CACHE:
        return _RAW_DTYPE_CACHE[path]

    size = os.path.getsize(path)
    cands: list[np.dtype] = []
    if size % 2 == 0:
        cands.append(np.uint16)
    if size % 4 == 0:
        cands.append(np.uint32)
    if not cands:
        raise AssertionError(f"{path}: size {size} not divisible by 2 or 4; not a raw token stream?")

    with open(path, "rb") as f:
        buf = f.read(min(sample_bytes, size))

    choice: Optional[np.dtype] = None
    if np.uint16 in cands:
        sample = np.frombuffer(buf[: (len(buf) // 2) * 2], dtype=np.uint16)
        if sample.size == 0 or sample.max() < 65536:
            choice = np.uint16

    if choice is None and np.uint32 in cands:
        sample = np.frombuffer(buf[: (len(buf) // 4) * 4], dtype=np.uint32)
        choice = np.uint32

    assert choice is not None
    _RAW_DTYPE_CACHE[path] = choice
    return choice


def _peek_data_shard(path: str) -> int:
    fmt, _ver = _detect_format(path)
    if fmt == "fineweb":
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
        return int(header[2])
    if fmt == "npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 1:
            raise AssertionError(f"{path}: npy shard must be 1D, got {arr.shape}")
        return int(arr.size)
    dtype = _inspect_raw_dtype(path)
    return os.path.getsize(path) // np.dtype(dtype).itemsize


def _load_data_shard(path: str) -> np.ndarray:
    fmt, _ver = _detect_format(path)

    if fmt == "fineweb":
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32, count=256)
            ntok = int(header[2])
            cur = f.tell()
            f.seek(0, os.SEEK_END)
            total = f.tell()
            f.seek(cur)
            data_bytes = total - cur
            if data_bytes == ntok * 2:
                dtype = np.uint16
            elif data_bytes == ntok * 4:
                dtype = np.uint32
            else:
                raise AssertionError(f"{path}: cannot infer dtype (data_bytes={data_bytes}, ntok={ntok})")
            tokens = np.frombuffer(f.read(), dtype=dtype)

    elif fmt == "npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        tokens = np.asarray(arr)

    else:
        dtype = _inspect_raw_dtype(path)
        tokens = np.memmap(path, mode="r", dtype=dtype)

    # Ensure uint16 token ids. Fail loudly if token ids too big.
    if tokens.dtype != np.uint16:
        sample = np.asarray(tokens[:1_000_000])
        if sample.size == 0 or sample.max() < 65536:
            tokens = np.asarray(tokens, dtype=np.uint16)
        else:
            raise AssertionError(f"{path}: token ids exceed 65535 (max={int(sample.max())})")

    return np.asarray(tokens)


class DistributedDataLoader:
    """
    Shard-cycling, rank-sharded token stream.
    More efficient version:
      - Reuses a persistent pinned CPU buffer (uint16) to avoid per-batch pin allocs
      - Avoids CPU dtype conversion; cast happens on GPU
      - Keeps async H2D (non_blocking=True)
    """

    def __init__(self, filename_pattern: str, B: int, T: int, rank: int, world_size: int, device: str = "cuda"):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.B = int(B)
        self.T = int(T)
        self.device = device

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files matching: {filename_pattern}"

        need = self.world_size * self.B * self.T + 1
        ntok_total = 0
        for fn in self.files:
            ntok = _peek_data_shard(fn)
            assert ntok >= need, f"{fn}: shard too small ({ntok}) for need={need}"
            ntok_total += int(ntok)
        self.ntok_total = int(ntok_total)

        self._cpu_buf_u16 = torch.empty((self.B * self.T + 1,), dtype=torch.uint16, pin_memory=True)

        self.current_shard = -1
        self.current_position = 0
        self.tokens: np.ndarray = np.empty((0,), dtype=np.uint16)
        self.advance()

    def reset(self) -> None:
        self.current_shard = -1
        self.advance()

    def advance(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        if not self.tokens.flags["C_CONTIGUOUS"]:
            self.tokens = np.ascontiguousarray(self.tokens)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, W = self.B, self.T, self.world_size
        global_bt = W * B * T

        if self.current_position + global_bt + 1 >= len(self.tokens):
            self.advance()

        rank_off = self.current_position + self.rank * (B * T)
        buf_np = self.tokens[rank_off : rank_off + (B * T + 1)]
        self.current_position += global_bt

        self._cpu_buf_u16.copy_(torch.from_numpy(buf_np), non_blocking=False)

        # Async H2D; cast on GPU (usually cheaper than CPU astype + extra copies)
        x = self._cpu_buf_u16[:-1].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        y = self._cpu_buf_u16[1:].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y

# ==================== ARGUMENT PARSING ====================

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 training with Value Mix architecture (v1 mixing + skip connections)")
    
    # Data paths
    parser.add_argument('--train_pattern', type=str, required=True,
                       help='Training data file pattern')
    parser.add_argument('--val_pattern', type=str, default='',
                       help='Validation data file pattern')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    
    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocabulary size')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    
    # Batching
    parser.add_argument('--batch_size', type=int, default=64, help='Device batch size (sequences per GPU)')
    parser.add_argument('--sequence_length', type=int, default=1024, help='Sequence length in tokens')
    parser.add_argument('--total_batch_size', type=int, default=524288, help='Total batch size in tokens (512*1024)')
    
    # Optimizer learning rates (4-optimizer setup)
    parser.add_argument('--embed_lr', type=float, default=0.3, help='Adam LR for embeddings (wte)')
    parser.add_argument('--head_lr', type=float, default=0.002, help='Adam LR for output head (lm_head)')
    parser.add_argument('--muon_lr', type=float, default=0.02, help='Muon LR for matrix parameters')
    parser.add_argument('--scalar_lr', type=float, default=0.02, help='Adam LR for scalar parameters (lambdas)')
    
    # Muon momentum warmup
    parser.add_argument('--muon_momentum_init', type=float, default=0.85, help='Initial Muon momentum')
    parser.add_argument('--muon_momentum_final', type=float, default=0.95, help='Final Muon momentum')
    parser.add_argument('--muon_momentum_warmup_steps', type=int, default=500, help='Steps for Muon momentum warmup')
    
    # LR schedule
    parser.add_argument('--warmup_frac', type=float, default=0.0, help='Fraction of steps for LR warmup')
    parser.add_argument('--warmdown_frac', type=float, default=0.03, help='Fraction of steps for LR warmdown')
    
    # Training
    parser.add_argument('--num_iterations', type=int, default=50000, help='Maximum training iterations')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    parser.add_argument("--val_every_steps", type=int, default=100,
                       help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    parser.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                       help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")
    parser.add_argument('--val_tokens', type=int, default=10485760, help='Tokens to use for validation')

    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    parser.add_argument("--checkpoint_every_steps", type=int, default=400,
                       help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    parser.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                       help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")

    # Stopping criteria (mutually exclusive modes)
    parser.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                       help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    parser.add_argument('--loss_threshold', type=float, default=3.3,
                       help='[const_loss mode] Stop when val_loss <= threshold (set to 0 to disable)')
    parser.add_argument("--stop_epoch_frac", type=float, default=None,
                       help="[epoch mode] Stop at this fraction of an epoch (e.g., 0.5). Required if stop_mode='epoch'.")
    
    # Numerics
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'],
                       help='Training dtype')
    parser.add_argument('--compile', action='store_true', default=True, help='Use torch.compile')
    parser.add_argument('--no-compile', dest='compile', action='store_false', help='Disable torch.compile')
    parser.add_argument('--tensorcores', action='store_true', default=True, help='Use tensor cores')
    parser.add_argument('--flash', type=int, default=1, help='Flash attention version')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # ==================== HELPER FUNCTIONS ====================

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

    def save_training_checkpoint(step, raw_model, optimizers, out_dir, val_loss=None):
        os.makedirs(out_dir, exist_ok=True)
        obj = {
            "step": step,
            "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            "optimizer1": optimizers[0].state_dict(),
            "optimizer2": optimizers[1].state_dict(),
            "optimizer3": optimizers[2].state_dict(),
            "optimizer4": optimizers[3].state_dict(),
            "val_loss": val_loss,
        }
        fname = f"checkpoint_step{step:06d}.pt"
        tmp_path = os.path.join(out_dir, fname + ".tmp")
        final_path = os.path.join(out_dir, fname)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)
        return final_path

    # ==================== SETUP ====================

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

    # ==================== LOGGING SETUP ====================

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
            "timestamp", "kind", "step", "epoch", "epoch_frac",
            "train_loss", "val_loss", 
            "lr_embed", "lr_head", "lr_muon", "lr_scalar", "muon_momentum",
            "grad_norm", "step_time_ms", "tokens_per_sec",
            "train_tokens", "val_tokens", "total_tokens",
            "train_time_s", "wall_time_s",
            "peak_mem_mib", "reserved_mem_mib", "free_mem_mib"
        ])
        csv_writer.writeheader()

        with open(logfile, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

    def log_metrics(row):
        if csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()

    def print0(*args_print, **kwargs):
        if master_process:
            msg = " ".join(str(a) for a in args_print)
            print(msg, **kwargs)
            if logfile:
                with open(logfile, "a") as f:
                    f.write(msg + "\n")

    # ==================== CALCULATE WARMUP/WARMDOWN STEPS ====================

    total_steps = args.num_iterations
    warmup_steps = int(round(args.warmup_frac * total_steps))
    warmdown_steps = int(round(args.warmdown_frac * total_steps))
    warmup_steps = max(0, warmup_steps)
    warmdown_steps = max(0, warmdown_steps)

    # Ensure warmup + warmdown doesn't exceed total_steps
    if warmup_steps + warmdown_steps > total_steps:
        warmdown_steps = max(0, total_steps - warmup_steps)

    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)

    # ==================== PRINT CONFIGURATION ====================

    print0(f"\n{'='*60}")
    print0("CONFIGURATION: VALUE MIX (7_value_mix)")
    print0(f"{'='*60}")
    print0(f"Device: {device} | DDP: {ddp} | World Size: {ddp_world_size}")
    print0(f"\nBatching:")
    print0(f"  Device batch size: {args.batch_size} sequences")
    print0(f"  Sequence length: {args.sequence_length} tokens")
    print0(f"  Total batch size: {args.total_batch_size:,} tokens")

    B, T = args.batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"  Gradient accumulation steps: {grad_accum_steps}")

    print0(f"\nOptimization (4 Optimizers - Value Mix Architecture):")
    print0(f"  1. Adam (wte/embeddings): lr={args.embed_lr}, betas=(0.9, 0.95)")
    print0(f"  2. Adam (lm_head/output): lr={args.head_lr}, betas=(0.9, 0.95)")
    print0(f"  3. Muon (matrix params): lr={args.muon_lr}, momentum={args.muon_momentum_init}→{args.muon_momentum_final}")
    print0(f"  4. Adam (scalar params): lr={args.scalar_lr}, betas=(0.9, 0.95)")

    print0(f"\nLR Schedule:")
    print0(f"  Total steps: {total_steps:,}")
    print0(f"  Warmup: {warmup_steps:,} steps ({args.warmup_frac:.1%})")
    print0(f"  Plateau: {plateau_steps:,} steps ({plateau_steps/total_steps:.1%})")
    print0(f"  Warmdown: {warmdown_steps:,} steps ({args.warmdown_frac:.1%})")
    print0(f"  Muon momentum warmup: {args.muon_momentum_warmup_steps} steps")

    print0(f"\nTraining:")
    print0(f"  Max iterations: {args.num_iterations:,}")
    print0(f"  Loss threshold (early stop): {args.loss_threshold}")
    print0(f"  Validation every: {args.val_every_steps} steps")
    print0(f"  Checkpoint every: {args.checkpoint_every_steps} steps")
    print0(f"{'='*60}\n")

    # ==================== MODEL & DATA ====================

    def _sync_device():
        if device_type == 'cuda':
            torch.cuda.synchronize()

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    globals()['FLASH'] = args.flash
    enc = tiktoken.get_encoding("gpt2")

    # Model initialization - use args for architecture
    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.sequence_length
    )
    model = GPT(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print0(f"  Architecture: Value Mix (v1 mixing, skip connections, lambda params)")

    model.train().to(device)
    if args.compile:
        print0("Compiling model with torch.compile()...")
        if inductor_config:
            inductor_config.coordinate_descent_tuning = True
        model = torch.compile(model)

    # Data loaders - use args for file patterns
    train_loader = DistributedDataLoader(args.train_pattern, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, ddp_rank, ddp_world_size) if args.val_pattern else None

    total_train_tokens = train_loader.ntok_total
    tokens_per_step = args.total_batch_size
    epoch_tokens = total_train_tokens
    epoch_steps = max(1, epoch_tokens // tokens_per_step)
    total_epochs = args.num_iterations / epoch_steps

    print0(f"\nData:")
    print0(f"  Train shards: {len(train_loader.files)} | Total tokens: {total_train_tokens:,}")
    if val_loader:
        print0(f"  Val shards: {len(val_loader.files)} | Total tokens: {val_loader.ntok_total:,}")
    print0(f"  Steps per epoch: ~{epoch_steps:,}")
    print0(f"  Total epochs: ~{total_epochs:.2f}")

    assert args.val_tokens % (B * T) == 0
    args.val_max_steps = args.val_tokens // (B * T)
    print0(f"  Val steps per eval: {args.val_max_steps} ({args.val_tokens:,} tokens)")

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

    # ==================== OPTIMIZERS (4-GROUP SETUP) ====================

    print0(f"\nConfiguring 4-optimizer setup (value mix)...")

    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2]

    skip_w = getattr(raw_model, "skip_weights", None)
    if skip_w is not None:
        scalar_params.append(skip_w)
        print0(f"  Found skip_weights parameter")

    print0(f"  Matrix params: {len(matrix_params)} matrices ({sum(p.numel() for p in matrix_params):,} params)")
    print0(f"  Scalar params: {len(scalar_params)} scalars ({sum(p.numel() for p in scalar_params):,} params)")

    optimizer1 = torch.optim.Adam(
        [raw_model.transformer.wte.weight],
        lr=args.embed_lr,
        betas=(0.9, 0.95),
        fused=(device_type=='cuda')
    )

    optimizer2 = torch.optim.Adam(
        [raw_model.lm_head.weight],
        lr=args.head_lr,
        betas=(0.9, 0.95),
        fused=(device_type=='cuda')
    )

    optimizer3 = Muon(
        matrix_params,
        lr=args.muon_lr,
        momentum=args.muon_momentum_init
    )

    optimizer4 = torch.optim.Adam(
        scalar_params,
        lr=args.scalar_lr,
        betas=(0.9, 0.95),
        fused=(device_type=='cuda')
    )

    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    # ==================== LR SCHEDULER ====================

    def get_lr(it):
        """LR multiplier using calculated warmup/warmdown steps."""
        it = min(it, total_steps)

        # Warmup
        if warmup_steps > 0 and it < warmup_steps:
            return (it + 1) / warmup_steps

        # Plateau
        decay_start = warmup_steps + plateau_steps
        if warmdown_steps <= 0 or it < decay_start:
            return 1.0

        # Warmdown (linear decay to 0)
        decay_ratio = (total_steps - it) / max(1, warmdown_steps)
        return max(0.0, decay_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # ==================== TRAINING LOOP ====================

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    training_time_s = 0.0
    train_tokens = 0
    val_tokens = 0
    total_tokens = 0
    counted_steps = 0
    skip_warmup_steps = 10
    timings = []
    wall_t0 = time.time()
    step_t0 = time.perf_counter()

    step = 0
    stop_now = False
    saved_final_ckpt = False
    val_loss = float("nan")

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    free_mib, total_mib = get_free_total_mib()
    print0(f"\nGPU Memory: {get_peak_mem_mib()} MiB allocated, {free_mib}/{total_mib} MiB free/total")
    print0(f"\n{'='*60}")
    print0("TRAINING STARTED")
    print0(f"{'='*60}\n")

    while True:
        last_step = (step == args.num_iterations)

        epoch_idx = step // epoch_steps
        step_in_epoch = step % epoch_steps
        epoch_frac = step_in_epoch / max(1, epoch_steps)

        # ==================== VALIDATION ====================
        run_validation = (val_loader is not None) and (val_loss_every > 0) and (step % val_loss_every == 0 or last_step)

        if run_validation:
            _sync_device()
            training_time_s += time.perf_counter() - step_t0

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                    output = model(x_val, y_val)
                    if isinstance(output, tuple):
                        _, loss_v = output
                    else:
                        loss_v = output
                    val_loss_t += loss_v.detach()
                val_loss_t /= max(1, args.val_max_steps)
                if ddp:
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                val_loss = val_loss_t.item()

            val_batch_tokens = args.val_max_steps * B * T * ddp_world_size
            val_tokens += val_batch_tokens
            total_tokens = train_tokens + val_tokens

            lr_embed = optimizers[0].param_groups[0]["lr"]
            lr_head = optimizers[1].param_groups[0]["lr"]
            lr_muon = optimizers[2].param_groups[0]["lr"]
            lr_scalar = optimizers[3].param_groups[0]["lr"]
            muon_momentum_current = optimizers[2].param_groups[0]["momentum"]

            wall_time = time.time() - wall_t0
            toks_per_sec = (train_tokens / training_time_s) if training_time_s > 0 else 0
            peak_mem = get_peak_mem_mib()
            reserved_mem = get_reserved_mem_mib()
            free_mem, _ = get_free_total_mib()

            print0(f"\n{'─'*60}")
            print0(f"[VAL] Step {step:,}/{args.num_iterations:,} | Epoch {epoch_idx+1} ({epoch_frac:.1%})")
            print0(f"      Val Loss: {val_loss:.6f} | Target: {args.loss_threshold}")
            print0(f"      LR: embed={lr_embed:.3g}, head={lr_head:.4g}, muon={lr_muon:.3g}, scalar={lr_scalar:.3g}")
            print0(f"      Muon momentum: {muon_momentum_current:.4f}")
            print0(f"      Train Time: {training_time_s:.1f}s | Wall: {wall_time:.1f}s")
            print0(f"      Train Tokens: {train_tokens:,} ({train_tokens/1e9:.2f}B)")
            print0(f"      Val Tokens: {val_tokens:,} ({val_tokens/1e9:.2f}B)")
            print0(f"      Total Tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
            print0(f"      Throughput: {toks_per_sec:,.0f} tok/s")
            print0(f"      Memory: {peak_mem} MiB peak | {reserved_mem} MiB reserved | {free_mem} MiB free")
            print0(f"{'─'*60}\n")

            log_metrics({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "kind": "val", "step": step, "epoch": epoch_idx + 1, "epoch_frac": f"{epoch_frac:.4f}",
                "train_loss": "", "val_loss": f"{val_loss:.6f}",
                "lr_embed": f"{lr_embed:.6g}", "lr_head": f"{lr_head:.6g}",
                "lr_muon": f"{lr_muon:.6g}", "lr_scalar": f"{lr_scalar:.6g}",
                "muon_momentum": f"{muon_momentum_current:.4f}",
                "grad_norm": "", "step_time_ms": "", "tokens_per_sec": f"{toks_per_sec:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens, "total_tokens": total_tokens,
                "train_time_s": f"{training_time_s:.1f}", "wall_time_s": f"{wall_time:.1f}",
                "peak_mem_mib": peak_mem, "reserved_mem_mib": reserved_mem, "free_mem_mib": free_mem
            })

            # Stopping criteria - mutually exclusive modes
            if args.stop_mode == "const_loss":
                # Mode 1: Stop by constant loss threshold
                if val_loss < args.loss_threshold and step > 0:
                    print0(f"🎯 [EARLY STOP] Val loss {val_loss:.4f} < threshold {args.loss_threshold:.4f}")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        print0(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True
                    stop_now = True
            elif args.stop_mode == "epoch":
                # Mode 2: Stop by epoch fraction
                if (args.stop_epoch_frac is not None) and (epoch_frac >= args.stop_epoch_frac):
                    print0(f"🎯 [EPOCH STOP] Reached {epoch_frac:.3f} epochs (target: {args.stop_epoch_frac})")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        print0(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True
                    stop_now = True

            if last_step and master_process and not saved_final_ckpt:
                ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                print0(f"   Final checkpoint saved: {ckpt_path}")
                saved_final_ckpt = True

            model.train()
            step_t0 = time.perf_counter()

        if last_step or stop_now:
            if last_step and not stop_now:
                print0(f"\n✓ Reached max iterations: {args.num_iterations:,}")
            break

        # ==================== TRAINING STEP ====================
        model.train()
        _sync_device()
        t0 = time.perf_counter()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        lossf = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            is_last = (micro_step == grad_accum_steps - 1)
            if ddp:
                model.require_backward_grad_sync = is_last
            with ctx:
                output = model(x, y)
                if isinstance(output, tuple):
                    _, loss_tensor = output
                else:
                    loss_tensor = output
                (loss_tensor / grad_accum_steps).backward()
                lossf += (loss_tensor / grad_accum_steps).detach()
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        norm_val = float(norm) if norm is not None else -1.0

        # Momentum warmup for Muon
        frac = min(step / args.muon_momentum_warmup_steps, 1.0)
        muon_momentum_current = (1 - frac) * args.muon_momentum_init + frac * args.muon_momentum_final
        optimizers[2].param_groups[0]['momentum'] = muon_momentum_current

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        _sync_device()
        step_dt = time.perf_counter() - t0
        timings.append(step_dt)

        train_tokens += tokens_per_step
        total_tokens = train_tokens + val_tokens

        if step >= skip_warmup_steps:
            training_time_s += step_dt
            counted_steps += 1

        # ==================== LOGGING ====================
        if master_process:
            lr_embed = optimizers[0].param_groups[0]["lr"]
            lr_muon = optimizers[2].param_groups[0]["lr"]
            toks_now = tokens_per_step / max(step_dt, 1e-9)

            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"loss {lossf.item():.4f} │ "
                f"lr {lr_embed:.3g}/{lr_muon:.3g} │ "
                f"μ {muon_momentum_current:.3f} │ "
                f"{step_dt*1000:6.1f}ms │ "
                f"{toks_now/1000:5.1f}k tok/s │ "
                f"tokens {total_tokens/1e6:.1f}M"
            )
            if device_type == "cuda":
                log_str += f" │ mem {get_peak_mem_mib()}MiB"
            print0(log_str)

            log_metrics({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "kind": "train", "step": step + 1, "epoch": epoch_idx + 1, "epoch_frac": f"{epoch_frac:.4f}",
                "train_loss": f"{lossf.item():.6f}", "val_loss": "",
                "lr_embed": f"{lr_embed:.6g}", "lr_head": f"{optimizers[1].param_groups[0]['lr']:.6g}",
                "lr_muon": f"{lr_muon:.6g}", "lr_scalar": f"{optimizers[3].param_groups[0]['lr']:.6g}",
                "muon_momentum": f"{muon_momentum_current:.4f}",
                "grad_norm": "", "step_time_ms": f"{step_dt*1000:.1f}", "tokens_per_sec": f"{toks_now:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens, "total_tokens": total_tokens,
                "train_time_s": f"{training_time_s:.1f}", "wall_time_s": f"{time.time() - wall_t0:.1f}",
                "peak_mem_mib": get_peak_mem_mib(), "reserved_mem_mib": get_reserved_mem_mib(),
                "free_mem_mib": get_free_total_mib()[0]
            })

        if master_process and checkpoint_every > 0 and (step == 0 or (step + 1) % checkpoint_every == 0):
            ckpt_path = save_training_checkpoint(step + 1 if step > 0 else 0, raw_model, optimizers, args.output_dir, val_loss)
            print0(f"   💾 Checkpoint saved: {ckpt_path}")

        step += 1
        step_t0 = time.perf_counter()

    # ==================== TRAINING COMPLETE ====================

    print0(f"\n{'='*60}")
    print0("TRAINING COMPLETE")
    print0(f"{'='*60}")

    wall_time = time.time() - wall_t0
    toks_per_sec = (train_tokens / training_time_s) if training_time_s > 0 else 0

    print0(f"\nFinal Statistics:")
    print0(f"  Steps completed: {step:,}")
    print0(f"  Training tokens: {train_tokens:,} ({train_tokens/1e9:.3f}B)")
    print0(f"  Validation tokens: {val_tokens:,} ({val_tokens/1e9:.3f}B)")
    print0(f"  Total tokens consumed: {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print0(f"  Training time: {training_time_s:.1f}s")
    print0(f"  Wall time: {wall_time:.1f}s")
    print0(f"  Average throughput: {toks_per_sec:,.0f} tok/s")
    print0(f"  Final val loss: {val_loss:.6f}")

    if timings:
        recent = timings[-20:]
        print0(f"  Last {len(recent)} steps avg: {np.mean(recent)*1000:.1f}ms")

    if device_type == "cuda":
        print0(f"  Peak memory: {get_peak_mem_mib()} MiB")

    if csv_file:
        csv_file.close()
    if ddp:
        destroy_process_group()

    print0(f"\nOutput directory: {args.output_dir}")
    print0(f"{'='*60}\n")



if __name__ == "__main__":
    main()
