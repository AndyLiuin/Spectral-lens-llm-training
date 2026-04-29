import os, math, glob, struct, inspect, time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import tiktoken 

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

print(f"Running pytorch {torch.version.__version__}")

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

# GPU inventory
import torch

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

def apply_norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

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

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(apply_norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(apply_norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target):
        idx = idx.view(-1)
        target = target.view(-1)

        docs = (idx == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[q_idx] == docs[kv_idx]
          window_mask = q_idx - kv_idx < 1024
          return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        x = self.transformer.wte(idx[None]) 
        x = F.rms_norm(x, (x.size(-1),)) 
        x0 = x
        v1 = None

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return loss

import sys  # <--- Missing
from typing import Tuple, Literal, Optional # <--- Missing

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

import os, time, math, numpy as np, glob
import sys
import torch
from contextlib import nullcontext
import tiktoken
try:
    from torch._inductor import config as inductor_config
except ImportError:
    inductor_config = None
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import io
import csv
import json
import wandb

# Make sure this is set BEFORE wandb.init
os.environ["WANDB_NOTEBOOK_NAME"] = "9_fixed_window"

def get_args():
    import argparse
    p = argparse.ArgumentParser("GPT-2 Training with Fixed Window Flex Attention")

    # Data paths
    p.add_argument("--train_pattern", type=str, required=True,
                   help="Glob pattern for training data shards")
    p.add_argument("--val_pattern", type=str, default="",
                   help="Glob pattern for validation data shards")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for checkpoints, logs, metrics")

    # Model
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=768)

    # Batching
    p.add_argument("--batch_size", type=int, default=8,
                   help="Global batch size (sequences)")
    p.add_argument("--device_batch_size", type=int, default=1,
                   help="Sequences per GPU (must be 1 for flex attention)")
    p.add_argument("--sequence_length", type=int, default=65536)

    # Optimizer learning rates (4-optimizer setup)
    p.add_argument("--embed_lr", type=float, default=0.6)
    p.add_argument("--head_lr", type=float, default=0.008)
    p.add_argument("--muon_lr", type=float, default=0.04)
    p.add_argument("--scalar_lr", type=float, default=0.04)

    # Muon momentum warmup
    p.add_argument("--muon_momentum_init", type=float, default=0.85)
    p.add_argument("--muon_momentum_final", type=float, default=0.95)
    p.add_argument("--muon_momentum_warmup_steps", type=int, default=500)

    # LR schedule
    p.add_argument("--warmup_frac", type=float, default=0.0)
    p.add_argument("--warmdown_frac", type=float, default=0.1)

    # Training
    p.add_argument("--num_iterations", type=int, default=20000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                   help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    p.add_argument("--loss_threshold", type=float, default=3.3,
                   help="[const_loss mode] Early stop when val_loss < threshold (set to 0 to disable)")
    p.add_argument("--stop_epoch_frac", type=float, default=None,
                   help="[epoch mode] Stop at this epoch fraction (e.g., 0.5). Required if stop_mode='epoch'.")

    # Window
    p.add_argument("--window_size", type=int, default=1024,
                   help="Fixed window size in tokens")

    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--val_every_steps", type=int, default=100,
                   help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    p.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                   help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")
    p.add_argument("--val_tokens", type=int, default=10485760)

    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--checkpoint_every_steps", type=int, default=400,
                   help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    p.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                   help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")

    # Numerics
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tensorcores", action="store_true")
    p.add_argument("--use_cudnn_attn", action="store_true")

    # WandB
    p.add_argument("--wandb_project", type=str, default="gpt2-dynamics")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="9_fixed_window_gpt2")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=1)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=[],
                   help="Tags for wandb run")
    p.add_argument("--wandb_save_code", action="store_true",
                   help="Save source code to wandb")

    return p.parse_args()

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
        "optimizers": [opt.state_dict() for opt in optimizers],
        "val_loss": val_loss,
    }
    fname = f"checkpoint_step{step:06d}.pt"
    tmp_path = os.path.join(out_dir, fname + ".tmp")
    final_path = os.path.join(out_dir, fname)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path

def main():
    args = get_args()
    WINDOW_SIZE = args.window_size

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
            "lr_embed", "lr_head", "lr_muon", "lr_scalar",
            "muon_momentum",
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

    if warmup_steps + warmdown_steps > total_steps:
        warmdown_steps = max(0, total_steps - warmup_steps)

    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)

    # ==================== PRINT CONFIGURATION ====================

    print0(f"\n{'='*60}")
    print0("CONFIGURATION: FIXED WINDOW FLEX ATTENTION (9_fixed_window)")
    print0(f"{'='*60}")
    print0(f"Device: {device} | DDP: {ddp} | World Size: {ddp_world_size}")
    print0(f"\nBatching (Flex Attention - Single Long Sequences):")

    B, T = args.device_batch_size, args.sequence_length

    print0(f"  Device batch size: {B} sequence (single long sequence per GPU)")
    print0(f"  Sequence length: {T:,} tokens ({T/1024:.0f}K tokens)")
    print0(f"  Total batch size: {args.batch_size} sequences across all devices")

    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.batch_size % (B * ddp_world_size) == 0
    grad_accum_steps = args.batch_size // (B * ddp_world_size)
    total_batch_tokens = tokens_per_fwdbwd * grad_accum_steps

    print0(f"  Tokens per forward/backward: {tokens_per_fwdbwd:,}")
    print0(f"  Gradient accumulation steps: {grad_accum_steps}")
    print0(f"  Total tokens per step: {total_batch_tokens:,}")

    print0(f"\nOptimization (4 Optimizers - UNet Double-LR + Fixed Window):")
    print0(f"  1. Adam (wte/embeddings): lr={args.embed_lr} (2x standard), betas=(0.9, 0.95)")
    print0(f"  2. Adam (lm_head/output): lr={args.head_lr} (4x standard), betas=(0.9, 0.95)")
    print0(f"  3. Muon (matrix params): lr={args.muon_lr} (2x standard), momentum={args.muon_momentum_init}→{args.muon_momentum_final}")
    print0(f"  4. Adam (scalar params): lr={args.scalar_lr} (2x standard), betas=(0.9, 0.95)")

    print0(f"\nLR Schedule:")
    print0(f"  Total steps: {total_steps:,}")
    print0(f"  Warmup: {warmup_steps:,} steps ({args.warmup_frac:.1%})")
    print0(f"  Plateau: {plateau_steps:,} steps ({plateau_steps/total_steps:.1%})")
    print0(f"  Warmdown: {warmdown_steps:,} steps ({args.warmdown_frac:.1%})")
    print0(f"  Muon momentum warmup: {args.muon_momentum_warmup_steps} steps")

    print0(f"\nFixed Window Attention:")
    print0(f"  Window size: {WINDOW_SIZE} tokens")
    print0(f"  Using flex_attention with document_causal_mask")

    print0(f"\nTraining:")
    print0(f"  Max iterations: {args.num_iterations:,}")
    print0(f"  Loss threshold (early stop): {args.loss_threshold}")
    print0(f"  Validation every: {args.val_every_steps} steps")
    print0(f"  Checkpoint every: {args.checkpoint_every_steps} steps")
    print0(f"  CUDNN attention: {args.use_cudnn_attn}")
    print0(f"{'='*60}\n")

    # ==================== MODEL & DATA ====================

    def _sync_device():
        if device_type == 'cuda':
            torch.cuda.synchronize()

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    enc = tiktoken.get_encoding("gpt2")

    model_config = GPTConfig()
    model = GPT(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print0(f"  Architecture: UNet + Fixed Window Flex Attention ({WINDOW_SIZE} tokens)")

    model.train().to(device)

    # CUDNN Attention Configuration
    if args.use_cudnn_attn:
        try:
            from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
            enable_cudnn_sdp(True)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            print0("Enabled CUDNN attention (faster than Flash)")
        except Exception as e:
            print0(f"Warning: Could not enable CUDNN attention: {e}")

    # IMPORTANT: compile AFTER setting SDPA backend flags
    if args.compile:
        print0("Compiling model with torch.compile()...")
        if inductor_config:
            inductor_config.coordinate_descent_tuning = True
        model = torch.compile(model)

    train_loader = DistributedDataLoader(args.train_pattern, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, ddp_rank, ddp_world_size) if args.val_pattern else None

    total_train_tokens = train_loader.ntok_total
    tokens_per_step = total_batch_tokens
    epoch_tokens = total_train_tokens
    epoch_steps = max(1, epoch_tokens // tokens_per_step)
    total_epochs = args.num_iterations / epoch_steps

    print0(f"\nData:")
    print0(f"  Train shards: {len(train_loader.files)} | Total tokens: {total_train_tokens:,}")
    if val_loader:
        print0(f"  Val shards: {len(val_loader.files)} | Total tokens: {val_loader.ntok_total:,}")
    print0(f"  Steps per epoch: ~{epoch_steps:,}")
    print0(f"  Total epochs: ~{total_epochs:.2f}")

    assert args.val_tokens % (B * T * ddp_world_size) == 0
    args.val_max_steps = args.val_tokens // (B * T * ddp_world_size)
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

    # ==================== OPTIMIZERS (4-GROUP SETUP - DOUBLE LR) ====================

    print0(f"\nConfiguring 4-optimizer setup (UNet double-LR + fixed window)...")

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
        it = min(it, total_steps)

        if warmup_steps > 0 and it < warmup_steps:
            return (it + 1) / warmup_steps

        decay_start = warmup_steps + plateau_steps
        if warmdown_steps <= 0 or it < decay_start:
            return 1.0

        decay_ratio = (total_steps - it) / max(1, warmdown_steps)
        return max(0.0, decay_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # ==================== WANDB INIT (rank 0 only) ====================

    wandb_run = None
    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)

        os.environ.setdefault("WANDB_DIR", args.output_dir)

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=list(args.wandb_tags) if isinstance(args.wandb_tags, (list, tuple)) else [str(args.wandb_tags)],
            mode=args.wandb_mode,
            config=vars(args),
        )

        # Make charts sane: use our own step axis
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.define_metric("loss/*", step_metric="global_step")
        wandb.define_metric("lr/*", step_metric="global_step")
        wandb.define_metric("muon/*", step_metric="global_step")
        wandb.define_metric("perf/*", step_metric="global_step")
        wandb.define_metric("mem/*", step_metric="global_step")
        wandb.define_metric("tokens/*", step_metric="global_step")
        wandb.define_metric("progress/*", step_metric="global_step")

        if args.wandb_save_code:
            try:
                wandb.run.log_code(".")
            except Exception:
                pass

    # ==================== SAVE STEP-0 CHECKPOINT (rank 0 only) ====================
    if master_process:
        ckpt0_path = save_training_checkpoint(0, raw_model, optimizers, args.output_dir, val_loss=None)
        print0(f"   💾 Saved initial checkpoint (step 0): {ckpt0_path}")

        if wandb_run is not None:
            try:
                art0 = wandb.Artifact("checkpoint_step0", type="model")
                art0.add_file(ckpt0_path)
                wandb.log_artifact(art0)
            except Exception:
                pass

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
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
                    with ctx:
                        loss_v = model(xv, yv)
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

            # ---- wandb: validation log (rank 0 only)
            if master_process and wandb_run is not None:
                wandb.log({
                    "global_step": step,
                    "loss/val": val_loss,
                    "val/loss": val_loss,
                    "progress/epoch": epoch_idx + 1,
                    "progress/epoch_frac": epoch_frac,
                    "tokens/val": val_tokens,
                    "tokens/total": total_tokens,
                    "lr/embed": lr_embed,
                    "lr/head": lr_head,
                    "lr/muon": lr_muon,
                    "lr/scalar": lr_scalar,
                    "muon/momentum": muon_momentum_current,
                    "perf/tokens_per_sec_avg": toks_per_sec,
                    "mem/peak_mib": peak_mem,
                    "mem/reserved_mib": reserved_mem,
                    "mem/free_mib": free_mem,
                })

            # Stopping criteria - mutually exclusive modes
            if args.stop_mode == "const_loss":
                # Mode 1: Stop by constant loss threshold
                if (args.loss_threshold > 0) and (val_loss < args.loss_threshold) and (step > 0):
                    print0(f"🎯 [EARLY STOP] Val loss {val_loss:.4f} < threshold {args.loss_threshold:.4f}")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        print0(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True

                        if wandb_run is not None:
                            try:
                                art = wandb.Artifact("checkpoints", type="model")
                                art.add_file(ckpt_path)
                                wandb.log_artifact(art)
                            except Exception:
                                pass

                    stop_now = True
            elif args.stop_mode == "epoch":
                # Mode 2: Stop by epoch fraction
                epoch_frac_current = (train_tokens / train_loader.ntok_total) if train_loader.ntok_total > 0 else 0.0
                if (args.stop_epoch_frac is not None) and (epoch_frac_current >= args.stop_epoch_frac):
                    print0(f"🎯 [EPOCH STOP] Reached {epoch_frac_current:.3f} epochs (target: {args.stop_epoch_frac})")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        print0(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True

                        if wandb_run is not None:
                            try:
                                art = wandb.Artifact("checkpoints", type="model")
                                art.add_file(ckpt_path)
                                wandb.log_artifact(art)
                            except Exception:
                                pass

                    stop_now = True

            if last_step and master_process and not saved_final_ckpt:
                ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                print0(f"   Final checkpoint saved: {ckpt_path}")
                saved_final_ckpt = True

                if wandb_run is not None:
                    try:
                        art = wandb.Artifact("checkpoints", type="model")
                        art.add_file(ckpt_path)
                        wandb.log_artifact(art)
                    except Exception:
                        pass

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

        step_losses = []
        for gas in range(grad_accum_steps):
            # For flex attention: x,y are already (B=1, T=65536), no micro-batching needed
            if ddp:
                # Only sync gradients on the last accumulation step
                model.require_backward_grad_sync = (gas == grad_accum_steps - 1)

            with ctx:
                loss = model(x, y)
                (loss / grad_accum_steps).backward()
                step_losses.append(loss.detach())

            # Load next batch
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            norm_val = float(grad_norm) if grad_norm is not None else -1.0
        else:
            norm_val = -1.0

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

        if step_losses:
            step_loss_t = torch.stack(step_losses).mean()
        else:
            step_loss_t = torch.tensor(0.0, device=device)

        if ddp:
            dist.all_reduce(step_loss_t, op=dist.ReduceOp.AVG)
        train_loss_scalar = step_loss_t.item()

        # ==================== LOGGING ====================
        if master_process:
            lr_embed = optimizers[0].param_groups[0]["lr"]
            lr_head = optimizers[1].param_groups[0]["lr"]
            lr_muon = optimizers[2].param_groups[0]["lr"]
            lr_scalar = optimizers[3].param_groups[0]["lr"]
            toks_now = tokens_per_step / max(step_dt, 1e-9)

            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"loss {train_loss_scalar:.4f} │ "
                f"gnorm {norm_val:6.2f} │ "
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
                "train_loss": f"{train_loss_scalar:.6f}", "val_loss": "",
                "lr_embed": f"{lr_embed:.6g}", "lr_head": f"{lr_head:.6g}",
                "lr_muon": f"{lr_muon:.6g}", "lr_scalar": f"{lr_scalar:.6g}",
                "muon_momentum": f"{muon_momentum_current:.4f}",
                "grad_norm": f"{norm_val:.4f}" if norm_val >= 0 else "",
                "step_time_ms": f"{step_dt*1000:.1f}", "tokens_per_sec": f"{toks_now:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens, "total_tokens": total_tokens,
                "train_time_s": f"{training_time_s:.1f}", "wall_time_s": f"{time.time() - wall_t0:.1f}",
                "peak_mem_mib": get_peak_mem_mib(), "reserved_mem_mib": get_reserved_mem_mib(),
                "free_mem_mib": get_free_total_mib()[0]
            })

            # ---- wandb: train log (rank 0 only)
            if wandb_run is not None and ((step + 1) % args.wandb_log_every == 0):
                wandb.log({
                    "global_step": step + 1,
                    "loss/train": train_loss_scalar,
                    "train/loss": train_loss_scalar,
                    "train/grad_norm": norm_val if norm_val >= 0 else None,
                    "lr/embed": lr_embed,
                    "lr/head": lr_head,
                    "lr/muon": lr_muon,
                    "lr/scalar": lr_scalar,
                    "muon/momentum": muon_momentum_current,
                    "perf/step_time_ms": step_dt * 1000.0,
                    "perf/tokens_per_sec": toks_now,
                    "progress/epoch": epoch_idx + 1,
                    "progress/epoch_frac": epoch_frac,
                    "tokens/train": train_tokens,
                    "tokens/total": total_tokens,
                    "mem/peak_mib": get_peak_mem_mib(),
                    "mem/reserved_mib": get_reserved_mem_mib(),
                    "mem/free_mib": get_free_total_mib()[0],
                })

                if step == 0:
                    print0("W&B logging active: loss/train and loss/val should appear under Charts.")

        if master_process and checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
            ckpt_path = save_training_checkpoint(step + 1, raw_model, optimizers, args.output_dir, val_loss)
            print0(f"   💾 Checkpoint saved: {ckpt_path}")

            if wandb_run is not None:
                try:
                    art = wandb.Artifact("checkpoints", type="model")
                    art.add_file(ckpt_path)
                    wandb.log_artifact(art)
                except Exception:
                    pass

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

    # wandb finalize
    if master_process and wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    print0(f"\nOutput directory: {args.output_dir}")
    print0(f"{'='*60}\n")


if __name__ == "__main__":
    main()
