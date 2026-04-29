import os
import io
import sys
import time
import math
import glob
import json
import struct
import shutil
import pickle
import inspect
import random
import argparse
from typing import Optional, Tuple, List
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import tiktoken

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch._inductor import config as inductor_config
except ImportError:
    inductor_config = None

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def get_peak_mem_mib() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() // 1024 // 1024)
    return 0

def get_reserved_mem_mib() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_reserved() // 1024 // 1024)
    return 0

def get_free_total_mib() -> Tuple[int, int]:
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return int(free // 1024 // 1024), int(total // 1024 // 1024)
        except Exception:
            pass
    return 0, 0

def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        # Check if safe to cast
        sample = np.asarray(tokens[:1_000_000])
        if sample.max() < 65536:
            tokens = np.asarray(tokens, dtype=np.uint16)
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

def save_training_checkpoint(
    step: int,
    raw_model: nn.Module,
    optimizers, # list of optimizers
    output_dir: str,
    val_loss: Optional[float],
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # Handle single optimizer case if list is not passed properly or mixed
    opts_state = []
    if isinstance(optimizers, list):
        opts_state = [opt.state_dict() for opt in optimizers]
    else:
        opts_state = [optimizers.state_dict()]

    ckpt = {
        "step": int(step),
        "val_loss": None if val_loss is None else float(val_loss),
        "model": raw_model.state_dict(),
        "optimizers": opts_state,
        "rng": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "extra": extra or {},
    }
    path = os.path.join(output_dir, f"ckpt_step_{step:07d}.pt")
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    return path

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if (seq_len != self.seq_len_cached
                or self.cos_cached is None
                or self.cos_cached.device != x.device):
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)  # same device as inv_freq (moved by .to())
            self.cos_cached = freqs.cos().to(dtype=torch.bfloat16)
            self.sin_cached = freqs.sin().to(dtype=torch.bfloat16)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

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

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() 
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # Flash Attention (SDPA)
        y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 
        self.transformer.wte.weight = self.lm_head.weight

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        
        if zero_stage == 1:
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

def get_args():
    p = argparse.ArgumentParser("GPT-2 RoPE Training")

    # Data
    p.add_argument("--train_pattern", type=str, required=True, help="Glob pattern for training data shards")
    p.add_argument("--val_pattern", type=str, default="", help="Glob pattern for validation data shards")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs")

    # Model
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=768)
    
    # Batching
    # Defaulting to T=1024, B=128 (global), device_B=64 to match previous script physics
    p.add_argument("--batch_size", type=int, default=512)         # global sequences
    p.add_argument("--device_batch_size", type=int, default=64)   # micro batch size
    p.add_argument("--sequence_length", type=int, default=1024)
    p.add_argument("--total_batch_size", type=int, default=None)  # legacy override support if needed

    # Optimization
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_iterations", type=int, default=50000)
    p.add_argument("--warmup_iters", type=int, default=0)
    p.add_argument("--learning_rate_decay_frac", type=float, default=0.0)
    p.add_argument("--zero_stage", type=int, default=0)

    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--val_every_steps", type=int, default=100,
                   help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    p.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                   help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")
    p.add_argument("--val_tokens", type=int, default=10485760)

    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--checkpoint_every_steps", type=int, default=800,
                   help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    p.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                   help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")

    # Stopping criteria (mutually exclusive modes)
    p.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                   help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    p.add_argument("--loss_threshold", type=float, default=3.3,
                   help="[const_loss mode] Stop when val_loss <= threshold (set to 0 to disable)")
    p.add_argument("--stop_epoch_frac", type=float, default=None,
                   help="[epoch mode] Stop at this fraction of an epoch (e.g., 0.5). Required if stop_mode='epoch'.")

    # System
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--tensorcores", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # WandB
    p.add_argument("--wandb_project", type=str, default="gpt2-dynamics")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="4_rope_gpt2")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=1)

    return p.parse_args()

import csv
def main():
    args = get_args()

    # DDP Setup
    ddp = int(os.environ.get("RANK", "-1")) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        device_type = "cuda"
        args.zero_stage = args.zero_stage 
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = device_type
        if args.device: device = args.device
        args.zero_stage = 0

    set_seed(args.seed, ddp_rank)

    # Logging Setup
    logfile = None
    csv_file = None
    csv_writer = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # Clear log
        with open(logfile, "w") as f:
            f.write(f"Log started at {time.asctime()}\n")
        
        # Save Args
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        # CSV metrics
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "timestamp", "step", "train_loss", "val_loss",
            "lr", "grad_norm", "step_time_ms", "tokens_per_sec",
            "train_tokens", "total_tokens", "peak_mem_mib"
        ])
        csv_writer.writeheader()

    def log_print(*a, **k):
        if not master_process: return
        msg = " ".join(str(x) for x in a)
        print(msg, **k)
        if logfile:
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    def log_row(row):
        if csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()

    if master_process:
        log_print(f"Running pytorch {torch.__version__} | CUDA runtime: {torch.version.cuda}")
        log_print(f"Device: {device}")

    # DataType
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if args.tensorcores and device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # WandB
    wandb_run = None
    if master_process and args.wandb_mode != "disabled":
        if wandb is None:
            log_print("WARNING: wandb not installed; forcing --wandb_mode disabled")
            args.wandb_mode = "disabled"
        else:
            os.environ.setdefault("WANDB_DIR", args.output_dir)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                mode=args.wandb_mode,
            )

    # Batch sizes
    B = args.device_batch_size
    T = args.sequence_length
    global_batch_size_seq = args.batch_size
    assert global_batch_size_seq % (B * ddp_world_size) == 0, "Global batch size must be divisible by (device_batch_size * world_size)"
    grad_accum_steps = global_batch_size_seq // (B * ddp_world_size)
    tokens_per_step = global_batch_size_seq * T 
    
    if master_process:
        log_print(f"Batch config: Global B={global_batch_size_seq}, Device B={B}, Grad Accum={grad_accum_steps}, T={T}")
        log_print(f"Tokens per step: {tokens_per_step:,}")

    # Data Loaders
    train_loader = DistributedDataLoader(args.train_pattern, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, ddp_rank, ddp_world_size) if args.val_pattern else None

    # Val Max Steps calculation
    val_max_steps = 20 # default
    if val_loader:
        assert args.val_tokens % (B * T * ddp_world_size) == 0, "Val tokens must be divisible by global batch size in tokens"
        val_max_steps = args.val_tokens // (B * T * ddp_world_size)
    
    # Epoch calculation for stopping criteria
    total_train_tokens = train_loader.ntok_total
    epoch_steps = max(1, total_train_tokens // tokens_per_step)
    if master_process:
        log_print(f"Epoch: {epoch_steps:,} steps per epoch, {args.num_iterations / epoch_steps:.2f} total epochs")

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
    
    # Model Init
    model_config = GPTConfig(
        block_size=T,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(model_config).to(device)
    
    if args.compile:
        log_print("Compiling model...")
        if inductor_config:
            inductor_config.coordinate_descent_tuning = True
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type,
        zero_stage=args.zero_stage
    )

    # LR Schedule (Cosine)
    def get_lr(it):
        it = min(it, args.num_iterations)
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        decay_ratio = (it - args.warmup_iters) / max(1, args.num_iterations - args.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    # Training State
    step = 0
    training_time_s = 0.0
    train_tokens = 0
    val_tokens_count = 0
    skip_warmup_steps = 10
    
    x, y = train_loader.next_batch()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    # Save Init Checkpoint
    if master_process:
        p0 = save_training_checkpoint(0, raw_model, optimizer, args.output_dir, val_loss=None)
        log_print(f"💾 Saved initial checkpoint: {p0}")

    free_mib, total_mib = get_free_total_mib()
    log_print(f"\nGPU Memory: peak {get_peak_mem_mib()} MiB | free/total {free_mib}/{total_mib} MiB")
    log_print(f"{'='*60}\nTRAINING STARTED\n{'='*60}")

    # Sync
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    wall_t0 = time.time()

    while step < args.num_iterations:
        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        last_step = (step == args.num_iterations - 1)
        if val_loader and (val_loss_every > 0) and ((step % val_loss_every == 0) or last_step):
            model.eval()
            val_loader.reset()
            if device_type == "cuda": torch.cuda.synchronize()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(val_max_steps):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
                    # Note: Forward inside no_grad and cast
                    with ctx:
                        _, loss_v = model(xv, yv, return_logits=False)
                    val_loss_t += loss_v.detach()
                val_loss_t /= max(1, val_max_steps)
                if ddp:
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                val_loss = val_loss_t.item()
            
            # Log Val
            if master_process:
                val_batch_tokens = val_max_steps * B * T * ddp_world_size
                val_tokens_count += val_batch_tokens
                total_tokens_now = train_tokens + val_tokens_count
                
                # Mock window for display consistency
                window_tokens = T
                sw_blocks = int(T // 128)
                
                # Retrieve current LR
                curr_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - wall_t0
                
                avg_tps = (train_tokens / training_time_s) if training_time_s > 0 else 0.0
                peak = get_peak_mem_mib()
                reserved = get_reserved_mem_mib()
                free, total = get_free_total_mib()
                
                log_print(f"\n{'─'*70}")
                log_print(f"[VAL] step {step:,}/{args.num_iterations:,} | epoch {step // grad_accum_steps} (approx)")
                log_print(f"      window {window_tokens} tok = {sw_blocks} blocks (dense)")
                log_print(f"      val_loss {val_loss:.6f} | threshold {args.loss_threshold}")
                log_print(f"      lr adam_base={curr_lr:.4g} muon=N/A")
                log_print(f"      μ N/A | train_time {training_time_s:.1f}s | wall {elapsed:.1f}s")
                log_print(f"      tokens train {train_tokens:,} | val {val_tokens_count:,} | total {total_tokens_now:,}")
                log_print(f"      avg throughput {avg_tps:,.0f} tok/s")
                log_print(f"      mem peak {peak} MiB | reserved {reserved} MiB | free {free}/{total} MiB")
                log_print(f"{'─'*70}\n")
                
                if wandb_run:
                    wandb.log({"val/loss": val_loss, "global_step": step})
                
                # Stopping criteria - mutually exclusive modes
                current_epoch_frac = step / epoch_steps
                stop_now = False

                if args.stop_mode == "const_loss":
                    # Mode 1: Stop by constant loss threshold
                    if (args.loss_threshold > 0) and (val_loss < args.loss_threshold) and (step > 0):
                        log_print(f"🎯 EARLY STOP: val_loss {val_loss:.4f} < {args.loss_threshold}")
                        stop_now = True
                elif args.stop_mode == "epoch":
                    # Mode 2: Stop by epoch fraction
                    if (args.stop_epoch_frac is not None) and (current_epoch_frac >= args.stop_epoch_frac):
                        log_print(f"🎯 EPOCH STOP: Reached {current_epoch_frac:.3f} epochs (target: {args.stop_epoch_frac})")
                        stop_now = True

                if stop_now:
                    save_training_checkpoint(step, raw_model, optimizer, args.output_dir, val_loss)
                    break
            
            model.train()

        # ---------------------------------------------------------------------
        # Training Step
        # ---------------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        
        step_loss_acc = torch.tensor(0.0, device=device)
        
        t0 = time.perf_counter()
        
        for micro_step in range(grad_accum_steps):
            is_last = (micro_step == grad_accum_steps - 1)
            if ddp:
                model.require_backward_grad_sync = is_last
            
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # Scale loss for gradient accumulation
                # Note: 'loss' returned is mean over batch. 
                # We want backward on loss / grad_accum_steps.
                (loss / grad_accum_steps).backward()
                step_loss_acc += loss.detach() / grad_accum_steps
            
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if ddp:
            dist.all_reduce(step_loss_acc, op=dist.ReduceOp.AVG)
        
        # Clip Gradient
        grad_norm = -1.0
        if args.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            if gn is not None: grad_norm = float(gn)
        
        # LR Update
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        dt = time.perf_counter() - t0
        
        if step >= skip_warmup_steps:
            training_time_s += dt
        
        train_tokens += tokens_per_step
        total_tokens_now = train_tokens + val_tokens_count
        
        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------
        if master_process:
            tps = tokens_per_step / max(dt, 1e-9)
            step_loss = step_loss_acc.item()
            
            # Format: 
            # step 1/5000 | win 128 | loss 10.8258 | gnorm 0.02 | lr 0.008/0.04 | μ 0.850 | 818.5ms | 640.6k tok/s | tokens 11.0M | mem 70683MiB
            
            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"win {T:4d} │ "
                f"loss {step_loss:.4f} │ "
                f"gnorm {grad_norm:6.2f} │ "
                f"lr {lr:.6f} │ "
                f"μ ----  │ "
                f"{dt*1000:7.1f}ms │ "
                f"{tps/1000:7.1f}k tok/s │ "
                f"tokens {total_tokens_now/1e6:7.1f}M"
            )
            if device_type == "cuda":
                log_str += f" │ mem {get_peak_mem_mib()}MiB"
            log_print(log_str)
            
            log_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step + 1,
                "train_loss": f"{step_loss:.6f}",
                "val_loss": "",
                "lr": f"{lr:.6g}",
                "grad_norm": f"{grad_norm:.4f}",
                "step_time_ms": f"{dt*1000:.1f}",
                "tokens_per_sec": f"{tps:.0f}",
                "train_tokens": train_tokens,
                "total_tokens": total_tokens_now,
                "peak_mem_mib": get_peak_mem_mib()
            })
            
            if wandb_run and ((step + 1) % args.wandb_log_every == 0):
                wandb.log({
                    "global_step": step + 1,
                    "train/loss": step_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "perf/step_time_ms": dt*1000,
                    "perf/tokens_per_sec": tps
                })
            
            # Checkpoint
            if checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
                save_training_checkpoint(step + 1, raw_model, optimizer, args.output_dir, None)

        step += 1

    # End
    if master_process:
        log_print("Training Finished.")
        if csv_file: csv_file.close()
        if wandb_run: wandb.finish()
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
