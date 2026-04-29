import os, math, glob, struct, inspect, time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

# Redirect caches to scratch to avoid home-dir quota issues
_SCRATCH = "/nfs/roberts/scratch/pi_jks79/zl664"
os.environ.setdefault("HF_HOME", f"{_SCRATCH}/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", f"{_SCRATCH}/hf_datasets_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_SCRATCH}/hf_cache")
os.environ.setdefault("WANDB_DIR", f"{_SCRATCH}/.cache/wandb")
os.environ.setdefault("WANDB_CACHE_DIR", f"{_SCRATCH}/.cache/wandb")
os.environ.setdefault("XDG_CACHE_HOME", f"{_SCRATCH}/.cache")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

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

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

import sys

# Required: tiktoken
try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken is required. Install with: pip install tiktoken", file=sys.stderr)
    sys.exit(1)


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
        assert config.n_embd % config.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
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

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float()
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


# =============================================================================
# Streaming data loader (HuggingFace + tiktoken)
# =============================================================================

class StreamingDataLoader:
    """
    Streams and tokenizes text from a HuggingFace dataset for GPT-2 training.

    Two modes:
    - Training (cache_mode=False): Continuously streams, tokenizes, and buffers
      tokens. Automatically refills the buffer when it runs low.
    - Validation (cache_mode=True): Streams and caches a fixed number of tokens
      at init. reset() returns to the start for reproducible evaluation.
    """

    def __init__(
        self,
        hf_dataset: str,
        hf_config: str,
        hf_split: str,
        B: int,
        T: int,
        rank: int,
        world_size: int,
        buffer_tokens: int = 20_000_000,
        cache_mode: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        device: str = "cuda",
        epoch_tokens: Optional[int] = None,
    ):
        self.B = int(B)
        self.T = int(T)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.cache_mode = cache_mode
        self._buffer_target = int(buffer_tokens)
        self._hf_dataset = hf_dataset
        self._hf_config = hf_config
        self._hf_split = hf_split
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed

        # GPT-2 tokenizer
        self._enc = tiktoken.get_encoding("gpt2")
        self._eot = self._enc.eot_token  # 50256

        # Pinned CPU buffer for async H2D transfer
        self._cpu_buf_u16 = torch.empty((B * T + 1,), dtype=torch.uint16, pin_memory=True)

        # Token buffer
        self.tokens: np.ndarray = np.empty(0, dtype=np.uint16)
        self.current_position = 0
        self._stream = None

        # Source info for logging
        self.source_name = f"{hf_dataset}/{hf_config}"

        # Start streaming and fill initial buffer
        self._stream = self._create_stream()

        if rank == 0:
            mode_str = "val-cache" if cache_mode else "train-stream"
            print(f"[data:{mode_str}] Streaming from {self.source_name} "
                  f"(split={hf_split}, shuffle={shuffle})...", flush=True)

        self._fill_buffer(buffer_tokens)

        if rank == 0:
            print(f"[data:{mode_str}] Initial buffer: {len(self.tokens):,} tokens", flush=True)

        # ntok_total: for cache mode, the actual cached size;
        # for training, the nominal epoch size (for scheduling/logging)
        if cache_mode:
            self.ntok_total = len(self.tokens)
        else:
            self.ntok_total = epoch_tokens if epoch_tokens else len(self.tokens)

    def _create_stream(self):
        """Create a new HuggingFace streaming iterator."""
        from datasets import load_dataset
        ds = load_dataset(
            self._hf_dataset,
            name=self._hf_config,
            split=self._hf_split,
            streaming=True,
            trust_remote_code=False,
        )
        if self._shuffle:
            ds = ds.shuffle(seed=self._shuffle_seed, buffer_size=10_000)
        return iter(ds)

    def _fill_buffer(self, target_tokens: int) -> None:
        """Ensure at least target_tokens are available from current_position."""
        available = len(self.tokens) - self.current_position
        needed = target_tokens - available
        if needed <= 0:
            return

        t0 = time.time()
        chunks: list[np.ndarray] = []
        total_new = 0

        while total_new < needed:
            try:
                example = next(self._stream)
                text = example.get("text", "")
                if not text:
                    continue
                toks = self._enc.encode_ordinary(text)
                toks.append(self._eot)
                chunks.append(np.array(toks, dtype=np.uint16))
                total_new += len(toks)
            except StopIteration:
                if self.cache_mode:
                    break  # Dataset segment exhausted, use what we have
                # Restart stream with new seed for different ordering
                self._shuffle_seed += 1
                self._stream = self._create_stream()

        if not chunks:
            return

        new_data = np.concatenate(chunks)

        # Compact: keep only unconsumed tokens + new data
        if self.current_position > 0:
            remaining = self.tokens[self.current_position:]
            self.tokens = np.concatenate([remaining, new_data])
            self.current_position = 0
        else:
            self.tokens = np.concatenate([self.tokens, new_data])

        if not self.tokens.flags["C_CONTIGUOUS"]:
            self.tokens = np.ascontiguousarray(self.tokens)

        dt = time.time() - t0
        if self.rank == 0 and not self.cache_mode:
            print(
                f"[data] Buffer refill: +{total_new:,} tokens in {dt:.1f}s "
                f"({total_new / max(dt, 1e-9):,.0f} tok/s) | "
                f"buffer={len(self.tokens):,} tokens",
                flush=True,
            )

    def reset(self) -> None:
        """Reset position to start (used for validation re-evaluation)."""
        self.current_position = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, W = self.B, self.T, self.world_size
        global_bt = W * B * T
        needed = global_bt + 1
        available = len(self.tokens) - self.current_position

        if available < needed:
            if self.cache_mode:
                # Wrap around for validation
                self.current_position = 0
            else:
                self._fill_buffer(self._buffer_target)
                available = len(self.tokens) - self.current_position
                if available < needed:
                    raise RuntimeError(
                        f"Not enough tokens after refill: {available} < {needed}. "
                        f"Increase --buffer_tokens or check dataset connectivity."
                    )

        rank_off = self.current_position + self.rank * (B * T)
        buf_np = self.tokens[rank_off : rank_off + (B * T + 1)]
        self.current_position += global_bt

        self._cpu_buf_u16.copy_(torch.from_numpy(buf_np), non_blocking=False)

        # Use int64 for embedding indices (nn.Embedding requires torch.long)
        x = self._cpu_buf_u16[:-1].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        y = self._cpu_buf_u16[1:].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y


import os, time, math, numpy as np, glob
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
import sys

def get_args():
    import argparse
    p = argparse.ArgumentParser("GPT-2 Training with Muon Optimizer (Streaming)")

    # HuggingFace streaming
    p.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb",
                   help="HuggingFace dataset name")
    p.add_argument("--hf_config", type=str, default="sample-100BT",
                   help="HuggingFace dataset config/subset for training")
    p.add_argument("--hf_val_config", type=str, default=None,
                   help="HuggingFace dataset config for validation. "
                        "If not set, uses --hf_config with deterministic (unshuffled) ordering.")
    p.add_argument("--hf_split", type=str, default="train",
                   help="HuggingFace dataset split")
    p.add_argument("--buffer_tokens", type=int, default=20_000_000,
                   help="Number of tokens to buffer in memory for training (default: 20M)")
    p.add_argument("--epoch_tokens", type=int, default=100_000_000_000,
                   help="Tokens per 'epoch' for scheduling/logging (default: 100B for sample-100BT)")

    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for checkpoints, logs, metrics")
    p.add_argument("--model", type=str, default="d12")

    # Batching
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sequence_length", type=int, default=1024)
    p.add_argument("--total_batch_size", type=int, default=524288)

    # Training
    p.add_argument("--num_iterations", type=int, default=50000)
    p.add_argument("--inference_only", type=int, default=0)

    # Optimizer
    p.add_argument("--embed_lr", type=float, default=0.0036)
    p.add_argument("--muon_lr", type=float, default=0.00036)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # LR schedule
    p.add_argument("--warmup_frac", type=float, default=0.0)
    p.add_argument("--warmdown_frac", type=float, default=0.1)
    p.add_argument("--lr_schedule", type=str, default="linear")
    p.add_argument("--min_lr_ratio", type=float, default=0.0)

    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--val_every_steps", type=int, default=100,
                   help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    p.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                   help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")
    p.add_argument("--val_tokens", type=int, default=10485760)
    p.add_argument("--sample_every", type=int, default=0)

    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--checkpoint_every_steps", type=int, default=400,
                   help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    p.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                   help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")

    # Numerics
    p.add_argument("--tensorcores", type=int, default=1)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--flash", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--zero_stage", type=int, default=0)

    # Stopping criteria (mutually exclusive modes)
    p.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                   help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    p.add_argument("--loss_threshold", type=float, default=3.3,
                   help="[const_loss mode] Stop when val_loss <= threshold (set to 0 to disable)")
    p.add_argument("--stop_epoch_frac", type=float, default=None,
                   help="[epoch mode] Stop at this epoch fraction (e.g., 0.5). Required if stop_mode='epoch'.")

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

def save_training_checkpoint(step, raw_model, optimizers, out_dir, val_loss=None, extra=None, filename=None):
    os.makedirs(out_dir, exist_ok=True)
    obj = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
        "optimizers": [opt.state_dict() for opt in optimizers],
        "val_loss": val_loss,
    }
    if extra:
        obj.update(extra)
    fname = filename or f"checkpoint_step{step:06d}.pt"
    tmp_path = os.path.join(out_dir, fname + ".tmp")
    final_path = os.path.join(out_dir, fname)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path

def main():
    args = get_args()

    # ==================== SETUP ====================

    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # ==================== LOGGING SETUP ====================

    logfile = None
    csv_file = None
    csv_writer = None

    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "train.log")

        # Write args to JSON
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        # Initialize CSV metrics file
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "timestamp", "kind", "step", "epoch", "epoch_frac",
            "train_loss", "val_loss", "lr", "grad_norm",
            "step_time_ms", "tokens_per_sec",
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

    # ==================== PRINT CONFIGURATION ====================

    print0(f"\n{'='*60}")
    print0("CONFIGURATION")
    print0(f"{'='*60}")
    print0(f"Device: {device} | DDP: {ddp} | World Size: {ddp_world_size}")
    print0(f"Model: {args.model} | Dtype: {args.dtype}")
    print0(f"\nBatching:")
    print0(f"  Device batch size: {B} sequences")
    print0(f"  Sequence length: {T} tokens")
    print0(f"  Total batch size: {args.total_batch_size:,} tokens")

    tokens_per_fwdbwd = B * T * ddp_world_size
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"  Gradient accumulation steps: {grad_accum_steps}")
    print0(f"  Tokens per micro-step: {tokens_per_fwdbwd:,}")

    print0(f"\nOptimization:")
    print0(f"  AdamW LR (embeddings): {args.embed_lr}")
    print0(f"  Muon LR (hidden): {args.muon_lr}")
    print0(f"  Muon momentum: {args.muon_momentum}")
    print0(f"  Weight decay: {args.weight_decay}")
    print0(f"  Grad clip: {args.grad_clip}")

    print0(f"\nLR Schedule:")
    print0(f"  Warmup fraction: {args.warmup_frac}")
    print0(f"  Warmdown fraction: {args.warmdown_frac}")
    print0(f"  Schedule type: {args.lr_schedule}")
    print0(f"  Min LR ratio: {args.min_lr_ratio}")

    print0(f"\nTraining:")
    print0(f"  Max iterations: {args.num_iterations:,}")
    print0(f"  Loss threshold (early stop): {args.loss_threshold}")
    print0(f"  Validation: every {args.val_every_steps} steps (step-based) or {args.val_every_epoch_frac} epoch frac")
    print0(f"  Checkpoint: every {args.checkpoint_every_steps} steps (step-based) or {args.save_every_epoch_frac} epoch frac")
    print0(f"\nStreaming:")
    print0(f"  Dataset: {args.hf_dataset}")
    print0(f"  Config: {args.hf_config}")
    print0(f"  Val Config: {args.hf_val_config or args.hf_config}")
    print0(f"  Split: {args.hf_split}")
    print0(f"  Buffer tokens: {args.buffer_tokens:,}")
    print0(f"  Epoch tokens: {args.epoch_tokens:,}")
    print0(f"{'='*60}\n")

    # ==================== MODEL & DATA ====================

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if device_type == "cuda" else nullcontext()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    globals()['FLASH'] = args.flash
    enc = tiktoken.get_encoding("gpt2")

    model_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=6, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    model.train().to(device)
    if args.compile:
        if hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        print0("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Validation: cache a fixed set of tokens at startup (deterministic, no shuffle)
    val_cache_tokens = args.val_tokens + ddp_world_size * B * T + 1
    val_config = args.hf_val_config or args.hf_config

    val_loader = StreamingDataLoader(
        hf_dataset=args.hf_dataset,
        hf_config=val_config,
        hf_split=args.hf_split,
        B=B, T=T, rank=ddp_rank, world_size=ddp_world_size,
        buffer_tokens=val_cache_tokens,
        cache_mode=True,
        shuffle=False,
        device=device,
    )

    # Training: continuous streaming with shuffle
    train_loader = StreamingDataLoader(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        B=B, T=T, rank=ddp_rank, world_size=ddp_world_size,
        buffer_tokens=args.buffer_tokens,
        cache_mode=False,
        shuffle=True,
        shuffle_seed=42,
        device=device,
        epoch_tokens=args.epoch_tokens,
    )

    # Calculate epochs and tokens
    total_train_tokens = train_loader.ntok_total
    tokens_per_step = args.total_batch_size
    epoch_tokens = total_train_tokens
    epoch_steps = max(1, epoch_tokens // tokens_per_step)
    total_epochs = args.num_iterations / epoch_steps

    print0(f"\nData:")
    print0(f"  Train source: {train_loader.source_name} | Epoch tokens: {total_train_tokens:,}")
    print0(f"  Val source: {val_loader.source_name} | Cached tokens: {val_loader.ntok_total:,}")
    print0(f"  Steps per epoch: ~{epoch_steps:,}")
    print0(f"  Total epochs: ~{total_epochs:.2f}")

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

    args.val_max_steps = 20
    tokens_per_val_step = B * T
    args.val_max_steps = args.val_tokens // tokens_per_val_step
    print0(f"  Val steps per eval: {args.val_max_steps} ({args.val_tokens:,} tokens)")

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # ==================== OPTIMIZERS & SCHEDULERS ====================

    print0(f"\nConfiguring optimizers...")
    optimizer1 = torch.optim.AdamW(
        raw_model.lm_head.parameters(),
        lr=args.embed_lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=(device_type=='cuda')
    )
    optimizer2 = Muon(
        raw_model.transformer.h.parameters(),
        lr=args.muon_lr, momentum=args.muon_momentum
    )
    optimizers = [optimizer1, optimizer2]

    total_steps = args.num_iterations
    warmup_steps = int(round(args.warmup_frac * total_steps))
    warmdown_steps = int(round(args.warmdown_frac * total_steps))
    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)
    decay_start = warmup_steps + plateau_steps

    print0(f"  LR schedule: warmup={warmup_steps}, plateau={plateau_steps}, warmdown={warmdown_steps}")

    def lr_mult(step_idx):
        s = int(step_idx)
        if s >= total_steps:
            return float(args.min_lr_ratio)
        if warmup_steps > 0 and s < warmup_steps:
            return float((s + 1) / warmup_steps)
        if s < decay_start or warmdown_steps == 0:
            return 1.0
        t = (s - decay_start) / max(1, warmdown_steps)
        t = min(max(t, 0.0), 1.0)
        if args.lr_schedule == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            shape = 1.0 - t
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * shape

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_mult) for opt in optimizers]

    # ==================== TRAINING LOOP ====================

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    training_time_s = 0.0
    train_tokens = 0  # Tokens used for training
    val_tokens = 0    # Tokens used for validation
    total_tokens = 0  # Total tokens consumed
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

    # Initial memory stats
    free_mib, total_mib = get_free_total_mib()
    print0(f"\nGPU Memory: {get_peak_mem_mib()} MiB allocated, {free_mib}/{total_mib} MiB free/total")
    print0(f"\n{'='*60}")
    print0("TRAINING STARTED")
    print0(f"{'='*60}\n")

    while True:
        last_step = (step == args.num_iterations)

        # Current epoch info
        epoch_idx = step // epoch_steps
        step_in_epoch = step % epoch_steps
        epoch_frac = step_in_epoch / max(1, epoch_steps)

        # ==================== VALIDATION ====================
        run_validation = (val_loss_every > 0) and (step % val_loss_every == 0 or last_step)

        if run_validation:
            torch.cuda.synchronize() if device_type == "cuda" else None
            training_time_s += time.perf_counter() - step_t0

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                    _, loss_v = model(x_val, y_val, return_logits=False)
                    val_loss_t += loss_v.detach()
                val_loss_t /= max(1, args.val_max_steps)
                if ddp:
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                val_loss = val_loss_t.item()

            # Track validation tokens
            val_batch_tokens = args.val_max_steps * B * T * ddp_world_size
            val_tokens += val_batch_tokens
            total_tokens = train_tokens + val_tokens

            # Comprehensive validation logging
            current_lr = optimizers[0].param_groups[0]["lr"]
            wall_time = time.time() - wall_t0
            step_avg_ms = (training_time_s * 1000 / counted_steps) if counted_steps > 0 else 0
            toks_per_sec = (train_tokens / training_time_s) if training_time_s > 0 else 0
            peak_mem = get_peak_mem_mib()
            reserved_mem = get_reserved_mem_mib()
            free_mem, _ = get_free_total_mib()

            print0(f"\n{'─'*60}")
            print0(f"[VAL] Step {step:,}/{args.num_iterations:,} | Epoch {epoch_idx+1} ({epoch_frac:.1%})")
            print0(f"      Val Loss: {val_loss:.6f} | Target: {args.loss_threshold}")
            print0(f"      LR: {current_lr:.6g} | Train Time: {training_time_s:.1f}s | Wall: {wall_time:.1f}s")
            print0(f"      Train Tokens: {train_tokens:,} ({train_tokens/1e9:.2f}B)")
            print0(f"      Val Tokens: {val_tokens:,} ({val_tokens/1e9:.2f}B)")
            print0(f"      Total Tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
            print0(f"      Throughput: {toks_per_sec:,.0f} tok/s")
            print0(f"      Memory: {peak_mem} MiB peak | {reserved_mem} MiB reserved | {free_mem} MiB free")
            print0(f"{'─'*60}\n")

            log_metrics({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "kind": "val", "step": step, "epoch": epoch_idx + 1, "epoch_frac": f"{epoch_frac:.4f}",
                "train_loss": "", "val_loss": f"{val_loss:.6f}", "lr": f"{current_lr:.6g}", "grad_norm": "",
                "step_time_ms": "", "tokens_per_sec": f"{toks_per_sec:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens, "total_tokens": total_tokens,
                "train_time_s": f"{training_time_s:.1f}", "wall_time_s": f"{wall_time:.1f}",
                "peak_mem_mib": peak_mem, "reserved_mem_mib": reserved_mem, "free_mem_mib": free_mem
            })

            # Stopping criteria - mutually exclusive modes
            epoch_frac_global = (train_tokens / train_loader.ntok_total) if train_loader.ntok_total > 0 else 0.0

            if args.stop_mode == "const_loss":
                # Mode 1: Stop by constant loss threshold
                if (args.loss_threshold > 0) and (val_loss < args.loss_threshold) and (step > 0):
                    print0(f"[EARLY STOP] Val loss {val_loss:.4f} < threshold {args.loss_threshold:.4f}")
                    if master_process:
                        ckpt_path = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        print0(f"   Checkpoint saved: {ckpt_path}")
                        saved_final_ckpt = True
                    stop_now = True
            elif args.stop_mode == "epoch":
                # Mode 2: Stop by epoch fraction
                if (args.stop_epoch_frac is not None) and (epoch_frac_global >= args.stop_epoch_frac):
                    print0(f"[EPOCH STOP] Reached {epoch_frac_global:.3f} epochs (target: {args.stop_epoch_frac})")
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

        # ==================== STOPPING ====================
        if last_step or stop_now:
            if last_step and not stop_now:
                print0(f"\nReached max iterations: {args.num_iterations:,}")
            break

        # ==================== TRAINING STEP ====================
        model.train()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        lossf = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            is_last = (micro_step == grad_accum_steps - 1)
            if ddp:
                model.require_backward_grad_sync = is_last
            with ctx:
                _, loss_tensor = model(x, y, return_logits=False)
                (loss_tensor / grad_accum_steps).backward()
                lossf += (loss_tensor / grad_accum_steps).detach()
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        norm_val = float(norm) if norm is not None else -1.0

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_dt = time.perf_counter() - t0
        timings.append(step_dt)

        # Track training tokens
        train_tokens += tokens_per_step
        total_tokens = train_tokens + val_tokens

        if step >= skip_warmup_steps:
            training_time_s += step_dt
            counted_steps += 1

        # ==================== LOGGING ====================
        if master_process:
            current_lr = optimizers[0].param_groups[0]["lr"]
            toks_now = tokens_per_step / max(step_dt, 1e-9)
            toks_avg = (train_tokens / training_time_s) if training_time_s > 0 else 0
            wall_time = time.time() - wall_t0
            step_avg_ms = (training_time_s * 1000 / counted_steps) if counted_steps > 0 else 0

            # Compact training log
            log_str = (
                f"step {step+1:5d}/{args.num_iterations} | "
                f"loss {lossf.item():.4f} | "
                f"gnorm {norm_val:6.2f} | "
                f"lr {current_lr:.4g} | "
                f"{step_dt*1000:6.1f}ms | "
                f"{toks_now/1000:5.1f}k tok/s | "
                f"tokens {total_tokens/1e6:.1f}M"
            )
            if device_type == "cuda":
                log_str += f" | mem {get_peak_mem_mib()}MiB"
            print0(log_str)

            log_metrics({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "kind": "train", "step": step + 1, "epoch": epoch_idx + 1, "epoch_frac": f"{epoch_frac:.4f}",
                "train_loss": f"{lossf.item():.6f}", "val_loss": "", "lr": f"{current_lr:.6g}", "grad_norm": f"{norm_val:.4f}",
                "step_time_ms": f"{step_dt*1000:.1f}", "tokens_per_sec": f"{toks_now:.0f}",
                "train_tokens": train_tokens, "val_tokens": val_tokens, "total_tokens": total_tokens,
                "train_time_s": f"{training_time_s:.1f}", "wall_time_s": f"{wall_time:.1f}",
                "peak_mem_mib": get_peak_mem_mib(), "reserved_mem_mib": get_reserved_mem_mib(), "free_mem_mib": get_free_total_mib()[0]
            })

        # ==================== PERIODIC CHECKPOINT ====================
        if master_process and checkpoint_every > 0 and (step == 0 or (step + 1) % checkpoint_every == 0):
            ckpt_path = save_training_checkpoint(step + 1 if step > 0 else 0, raw_model, optimizers, args.output_dir, val_loss)
            print0(f"   Checkpoint saved: {ckpt_path}")

        step += 1
        step_t0 = time.perf_counter()

    # ==================== TRAINING COMPLETE ====================

    print0(f"\n{'='*60}")
    print0("TRAINING COMPLETE")
    print0(f"{'='*60}")

    # Final stats
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

    # Cleanup
    if csv_file:
        csv_file.close()
    if ddp:
        destroy_process_group()

    print0(f"\nOutput directory: {args.output_dir}")
    print0(f"{'='*60}\n")


if __name__ == "__main__":
    main()
