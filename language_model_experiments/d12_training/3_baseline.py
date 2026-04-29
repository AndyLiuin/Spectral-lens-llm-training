#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 small baseline with AdamW + DDP.
Cluster-ready with:
  - simple FineWeb shard reader
  - torchrun/SLURM-friendly DDP init
  - argparse CLI
  - out_dir/run_name logging + optional checkpointing
  - Iteration-based training: --num_iterations (stops after N optimizer steps)
  - Cosine LR schedule with warmup_iters

Example:
  torchrun --nproc_per_node=4 3_baseline.py \\
    --train_pattern "/path/fineweb_train_*.bin" \\
    --val_pattern   "/path/fineweb_val_*.bin" \\
    --out_dir "/path/runs/gpt2_small" \\
    --run_name "gpt2_small_baseline" \\
    --num_iterations 20000 --compile --bf16
"""

from __future__ import annotations

import os
import sys
import glob
import time
import uuid
import argparse
import csv
import json
import signal
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Small utilities
# =============================================================================

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

    # Ensure uint16 token ids (GPT-2-ish). Fail loudly if token ids too big.
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

        # Use int64 for embedding indices (nn.Embedding requires torch.long)
        x = self._cpu_buf_u16[:-1].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        y = self._cpu_buf_u16[1:].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # Squared ReLU
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: "GPTConfig"):
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
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_logits: bool = True):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

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
# CLI + main
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("gpt2_baseline_epoch")

    # paths
    p.add_argument("--train_pattern", type=str, required=True)
    p.add_argument("--val_pattern", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=768)

    # batching
    p.add_argument("--global_batch_size", type=int, default=512, help="in sequences, across all GPUs")
    p.add_argument("--device_batch_size", type=int, default=64, help="in sequences, per GPU")
    p.add_argument("--sequence_length", type=int, default=1024, help="tokens")

    # stopping (iteration-based)
    p.add_argument("--num_iterations", type=int, default=50000,
                   help="Total number of optimizer steps to train.")

    # optimizer hyperparams
    p.add_argument("--learning_rate", type=float, default=0.0006, help="Peak learning rate for AdamW")
    p.add_argument("--weight_decay", type=float, default=0.1)

    # LR schedule
    p.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup steps.")
    p.add_argument("--learning_rate_decay_frac", type=float, default=0.0,
                   help="Final LR as fraction of peak (0.0 => decay to zero).")

    # validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--val_every_steps", type=int, default=0,
                   help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    p.add_argument("--val_every_epoch_frac", type=float, default=0.025,
                   help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")

    # checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--checkpoint_every_steps", type=int, default=0,
                   help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    p.add_argument("--save_every_epoch_frac", type=float, default=0.1,
                   help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")

    # val set size
    p.add_argument("--val_tokens", type=int, default=10_485_760)

    # stopping criteria (mutually exclusive modes)
    p.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"],
                   help="Stopping mode: 'const_loss' stops when val_loss reaches threshold, 'epoch' stops after certain epochs.")
    p.add_argument("--loss_threshold", type=float, default=None,
                   help="[const_loss mode] Stop training when val_loss <= this threshold (e.g., 2.6). If None, train for full duration.")
    p.add_argument("--stop_epoch_frac", type=float, default=None,
                   help="[epoch mode] Stop at this fraction of an epoch (e.g., 0.5). Required if stop_mode='epoch'.")

    # perf toggles
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true")

    # wandb
    p.add_argument("--wandb", action="store_true", default=True, help="Enable wandb logging (default: True)")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--wandb_project", type=str, default="modded-nanogpt", help="Wandb project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team) name")

    return p


def main():
    args = build_argparser().parse_args()

    def now_str() -> str:
        try:
            return now_ts()
        except Exception:
            return time.strftime("%Y-%m-%d %H:%M:%S")

    def env_int(k: str, default: int) -> int:
        try:
            return int(os.environ.get(k, default))
        except Exception:
            return default

    def is_dist() -> bool:
        return dist.is_available() and dist.is_initialized()

    def barrier():
        if not is_dist():
            return
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()

    def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
        if is_dist() and dist.get_world_size() > 1:
            y = x.clone()
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            y /= dist.get_world_size()
            return y
        return x

    def get_peak_mem_mib() -> int:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated() // 1024 // 1024)

    def get_reserved_mem_mib() -> int:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_reserved() // 1024 // 1024)

    def get_free_total_mib() -> tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)
        try:
            free, total = torch.cuda.mem_get_info()
            return int(free // 1024 // 1024), int(total // 1024 // 1024)
        except Exception:
            return (0, 0)

    class RunLogger:
        """
        Master-only logger writing:
          - logs/train.log (human readable)
          - logs/metrics.csv (structured)
        """
        def __init__(self, run_dir: Path, enabled: bool):
            self.enabled = bool(enabled)
            self.run_dir = run_dir
            self.log_path = run_dir / "logs" / "train.log"
            self.csv_path = run_dir / "logs" / "metrics.csv"
            self._csv_f = None
            self._csv_w = None
            if self.enabled:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

        def log(self, msg: str, console: bool = True) -> None:
            if not self.enabled:
                return
            line = f"[{now_str()}] {msg}"
            if console:
                print(line, flush=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        def write_row(self, row: dict) -> None:
            if not self.enabled:
                return
            if self._csv_f is None:
                self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
                self._csv_w = csv.DictWriter(self._csv_f, fieldnames=list(row.keys()))
                if self._csv_f.tell() == 0:
                    self._csv_w.writeheader()
            assert self._csv_w is not None
            self._csv_w.writerow(row)
            self._csv_f.flush()

        def close(self) -> None:
            if self._csv_f is not None:
                try:
                    self._csv_f.close()
                finally:
                    self._csv_f = None
                    self._csv_w = None

    def snapshot_run(logger: RunLogger, args, run_dir: Path) -> None:
        if not logger.enabled:
            return

        # source
        try:
            code = read_self_code()
            with open(run_dir / "logs" / "source.py", "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logger.log(f"Warning: failed to write source.py: {e}", console=True)

        # args
        try:
            with open(run_dir / "logs" / "args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)
        except Exception as e:
            logger.log(f"Warning: failed to write args.json: {e}", console=True)

        # env
        try:
            env_dump = dict(
                python=sys.version,
                platform=platform.platform(),
                torch_version=torch.__version__,
                torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                hostname=os.uname().nodename if hasattr(os, "uname") else "",
                slurm=dict(
                    job_id=os.environ.get("SLURM_JOB_ID", ""),
                    node=os.environ.get("SLURMD_NODENAME", ""),
                    nodelist=os.environ.get("SLURM_NODELIST", ""),
                    procid=os.environ.get("SLURM_PROCID", ""),
                    localid=os.environ.get("SLURM_LOCALID", ""),
                ),
            )
            with open(run_dir / "logs" / "env.json", "w", encoding="utf-8") as f:
                json.dump(env_dump, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.log(f"Warning: failed to write env.json: {e}", console=True)

        logger.log("===== nvidia-smi =====", console=True)
        try:
            logger.log(nvidia_smi(), console=False)
        except Exception as e:
            logger.log(f"(nvidia-smi failed: {e})", console=True)

    # -------------------------------------------------------------------------
    # torchrun env
    # -------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Only initialize DDP if we're in a distributed environment (RANK is set)
    if "RANK" in os.environ and not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")

    barrier()

    set_seed(args.seed, rank)
    master = is_master(rank)

    out_root = Path(args.out_dir)
    run_tag = args.run_name.strip() or f"run_{uuid.uuid4().hex[:8]}"
    run_dir = out_root / run_tag

    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    barrier()

    logger = RunLogger(run_dir, enabled=master)
    
    # Wandb initialization (master only)
    use_wandb = master and WANDB_AVAILABLE and args.wandb and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_tag,
            config=vars(args),
            dir=str(run_dir),
        )
    
    if master:
        snapshot_run(logger, args, run_dir)
        free_mib, total_mib = get_free_total_mib()
        logger.log(
            f"Starting run. rank={rank} local_rank={local_rank} world_size={world_size} "
            f"device={device} cuda={torch.version.cuda} torch={torch.__version__} "
            f"mem_free/total={free_mib}/{total_mib} MiB",
            console=True,
        )
        logger.log(f"train_pattern={args.train_pattern}", console=True)
        logger.log(f"val_pattern={args.val_pattern}", console=True)
        if use_wandb:
            logger.log(f"wandb enabled: project={args.wandb_project}, run={run_tag}", console=True)

    # -------------------------------------------------------------------------
    # graceful termination (SIGTERM/SIGINT) + sync across ranks
    # -------------------------------------------------------------------------
    stop_requested = False

    def _handle_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)
    except Exception:
        pass

    def sync_stop_flag() -> bool:
        nonlocal stop_requested
        if not is_dist():
            return stop_requested
        flag = torch.tensor([1 if stop_requested else 0], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        stop_requested = (int(flag.item()) == 1)
        return stop_requested

    # -------------------------------------------------------------------------
    # batch sizing
    # -------------------------------------------------------------------------
    B = int(args.device_batch_size)
    T = int(args.sequence_length)

    denom_val = B * T * world_size
    if args.val_tokens % denom_val != 0:
        raise AssertionError(f"--val_tokens ({args.val_tokens}) must be divisible by B*T*world_size ({denom_val})")
    val_steps = args.val_tokens // denom_val

    denom_train = B * world_size
    if args.global_batch_size % denom_train != 0:
        raise AssertionError(
            f"--global_batch_size ({args.global_batch_size}) must be divisible by device_batch_size*world_size ({denom_train})"
        )
    grad_accum_steps = args.global_batch_size // denom_train
    if grad_accum_steps < 1:
        raise AssertionError("grad_accum_steps must be >= 1")

    tokens_per_step = int(args.global_batch_size) * int(T)
    cum_tokens = 0

    # -------------------------------------------------------------------------
    # data
    # -------------------------------------------------------------------------
    train_loader = DistributedDataLoader(args.train_pattern, B, T, rank, world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, rank, world_size)

    # ---------------- epoch definition ----------------
    # One "epoch" is a full pass through ALL train shard tokens (ntok_total),
    # but training advances in discrete steps of tokens_per_step.
    # We use floor() so epoch boundaries align to step boundaries.
    epoch_tokens = int(train_loader.ntok_total)
    epoch_steps = max(1, epoch_tokens // tokens_per_step)
    epoch_tokens_effective = epoch_steps * tokens_per_step

    # total steps from num_iterations
    total_steps = args.num_iterations

    # total tokens processed (effective, step-aligned)
    total_train_tokens = total_steps * tokens_per_step

    # validation cadence: step-based takes precedence over epoch-fraction-based
    if args.val_every_steps > 0:
        val_every_steps = args.val_every_steps
    elif args.val_every_epoch_frac > 0:
        val_every_steps = max(1, int(round(float(args.val_every_epoch_frac) * epoch_steps)))
    else:
        val_every_steps = 0

    # checkpoint cadence: step-based takes precedence over epoch-fraction-based
    if args.checkpoint_every_steps > 0:
        save_every_steps = args.checkpoint_every_steps
    elif args.save_every_epoch_frac > 0:
        save_every_steps = max(1, int(round(float(args.save_every_epoch_frac) * epoch_steps)))
    else:
        save_every_steps = 0

    if master:
        logger.log(f"Train shards: {len(train_loader.files)} | total_tokens={train_loader.ntok_total:,}", console=True)
        logger.log(f"Val shards:   {len(val_loader.files)} | total_tokens={val_loader.ntok_total:,}", console=True)
        logger.log(f"grad_accum_steps={grad_accum_steps} | per-rank batch={B} | seq_len={T}", console=True)
        logger.log(f"tokens_per_step={tokens_per_step:,}", console=True)
        logger.log(f"epoch_tokens(raw)={epoch_tokens:,} | epoch_steps={epoch_steps:,} | epoch_tokens(effective)={epoch_tokens_effective:,}", console=True)
        logger.log(f"Epoch: {epoch_steps:,} steps per epoch, {args.num_iterations / epoch_steps:.2f} total epochs", console=True)
        logger.log(f"num_iterations={total_steps:,}", console=True)
        logger.log(f"val_every_steps={val_every_steps:,} (epoch_frac={args.val_every_epoch_frac})", console=True)
        logger.log(f"save_every_steps={save_every_steps:,} (epoch_frac={args.save_every_epoch_frac})", console=True)
        logger.log(
            f"LR schedule: warmup_iters={args.warmup_iters}, "
            f"learning_rate_decay_frac={args.learning_rate_decay_frac}",
            console=True,
        )

    # -------------------------------------------------------------------------
    # model
    # -------------------------------------------------------------------------
    cfg = GPTConfig(
        vocab_size=int(args.vocab_size),
        block_size=int(args.block_size),
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_embd=int(args.n_embd),
    )
    model = GPT(cfg).to(device)

    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True

    if args.compile:
        if master:
            logger.log("Compiling model with torch.compile(...)", console=True)
        model = torch.compile(model)

    if is_dist():
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        raw_model = ddp_model.module
    else:
        ddp_model = model
        raw_model = model

    # -------------------------------------------------------------------------
    # optimizer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(args.weight_decay),
        fused=True,
    )

    # -------------------------------------------------------------------------
    # LR schedule: cosine decay with warmup (matching 4_rope_fineweb.py)
    # -------------------------------------------------------------------------
    import math

    def get_lr(it: int) -> float:
        it = min(it, args.num_iterations)
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        decay_ratio = (it - args.warmup_iters) / max(1, args.num_iterations - args.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    # autocast
    def autocast_ctx():
        if args.bf16:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext
        return nullcontext()

    # -------------------------------------------------------------------------
    # timing and metrics
    # -------------------------------------------------------------------------
    training_time_ms = 0.0
    total_t0 = time.perf_counter()
    step_t0 = time.perf_counter()

    warm_ignore = 10
    steps_counted = 0
    sum_step_time = 0.0

    torch.cuda.reset_peak_memory_stats()

    # prefetch first batch
    train_loader.reset()
    x, y = train_loader.next_batch()

    val_loss_scalar = float("nan")
    target_reached = False  # flag to indicate if val_loss_target was reached

    # wall clock for timing
    t0_wall = time.time()

    # -------------------------------------------------------------------------
    # train loop
    # -------------------------------------------------------------------------
    try:
        step = 0
        while step < args.num_iterations:
            last_step = (step == args.num_iterations - 1)

            # sync stop request across ranks
            if sync_stop_flag():
                last_step = True

            # reset timing window after warm_ignore
            if step == warm_ignore:
                training_time_ms = 0.0
                torch.cuda.synchronize()
                t0_wall = time.time()
                step_t0 = time.perf_counter()
                steps_counted = 0
                sum_step_time = 0.0

            timed_steps = float("nan") if step <= (warm_ignore + 1) else (step - warm_ignore) + 1

            # ---------------- epoch progress ----------------
            epoch_idx = step // epoch_steps
            step_in_epoch = step % epoch_steps
            epoch_frac = (step_in_epoch / epoch_steps) if epoch_steps > 0 else 0.0
            effective_epochs = (step / epoch_steps) if epoch_steps > 0 else 0.0

            # ---------------- validation ----------------
            do_val = last_step or (val_every_steps > 0 and step % val_every_steps == 0)
            if do_val:
                barrier()
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.time() - t0_wall)

                ddp_model.eval()
                val_loader.reset()
                val_loss = torch.zeros((), device=device, dtype=torch.float32)

                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    with autocast_ctx():
                        _, loss = ddp_model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach().float()
                    del loss

                val_loss /= float(val_steps)
                val_loss = all_reduce_mean(val_loss)
                val_loss_scalar = float(val_loss.item())

                # Stopping criteria - mutually exclusive modes
                if args.stop_mode == "const_loss":
                    # Mode 1: Stop by constant loss threshold
                    if args.loss_threshold is not None and val_loss_scalar <= args.loss_threshold:
                        target_reached = True
                        last_step = True
                        if master:
                            logger.log(
                                f"[target reached] val_loss {val_loss_scalar:.4f} <= threshold {args.loss_threshold:.4f}. Stopping training.",
                                console=True,
                            )
                elif args.stop_mode == "epoch":
                    # Mode 2: Stop by epoch fraction
                    if args.stop_epoch_frac is not None and epoch_frac >= args.stop_epoch_frac:
                        target_reached = True
                        last_step = True
                        if master:
                            logger.log(
                                f"[epoch target] Reached {epoch_frac:.3f} epoch fraction (target: {args.stop_epoch_frac}). Stopping training.",
                                console=True,
                            )

                ddp_model.train()
                barrier()
                torch.cuda.synchronize()
                t0_wall = time.time()
                step_t0 = time.perf_counter()

                if master:
                    total_time_s = time.perf_counter() - total_t0
                    peak = get_peak_mem_mib()
                    rsv = get_reserved_mem_mib()
                    free_mib, total_mib = get_free_total_mib()
                    logger.log(
                        f"[val] step {step}/{args.num_iterations} "
                        f"(epoch {epoch_idx + 1} + {epoch_frac:.2%}) "
                        f"val_loss {val_loss_scalar:.4f} | "
                        f"train_time {training_time_ms/1000.0:.1f}s total_time {total_time_s:.1f}s | "
                        f"mem peak_alloc {peak} MiB reserved {rsv} MiB free/total {free_mib}/{total_mib} MiB"
                        + (f" | TARGET: {args.loss_threshold}" if args.loss_threshold is not None else ""),
                        console=True,
                    )
                    logger.write_row(
                        dict(
                            ts=now_str(),
                            kind="val",
                            step=step,
                            epoch=float(effective_epochs),
                            train_loss=float("nan"),
                            val_loss=val_loss_scalar,
                            lr=float(optimizer.param_groups[0]["lr"]),
                            train_time_s=training_time_ms / 1000.0,
                            total_time_s=total_time_s,
                            step_time_s=float("nan"),
                            toks_now=float("nan"),
                            toks_avg=float("nan"),
                            cum_tokens=cum_tokens,
                            peak_mem_mib=peak,
                            reserved_mem_mib=rsv,
                            free_mem_mib=free_mib,
                            total_mem_mib=total_mib,
                        )
                    )
                    if use_wandb:
                        wandb.log({
                            "val_loss": val_loss_scalar,
                            "epoch": float(effective_epochs),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "train_time_s": training_time_ms / 1000.0,
                            "cum_tokens": cum_tokens,
                            "peak_mem_mib": peak,
                        }, step=step)

            # ---------------- checkpoint ----------------
            do_save = last_step or (save_every_steps > 0 and (step % save_every_steps == 0))

            if do_save and master:
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.time() - t0_wall)

                ckpt = dict(
                    step=step,
                    epoch=float(effective_epochs),
                    args=vars(args),
                    derived=dict(
                        total_steps=total_steps,
                        epoch_steps=epoch_steps,
                        epoch_tokens=epoch_tokens,
                        epoch_tokens_effective=epoch_tokens_effective,
                        tokens_per_step=tokens_per_step,
                        warmup_steps=args.warmup_iters,
                        warmdown_steps=0,
                    ),
                    model=raw_model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    val_loss=val_loss_scalar,
                    target_reached=target_reached,
                    rng=dict(
                        torch=torch.random.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all(),
                        numpy=np.random.get_state(),
                    ),
                )
                # Use different naming if target was reached
                if target_reached:
                    ckpt_path = run_dir / "checkpoints" / f"checkpoint_target_reached_step{step:06d}.pt"
                else:
                    ckpt_path = run_dir / "checkpoints" / f"checkpoint_step{step:06d}.pt"
                torch.save(ckpt, ckpt_path)
                logger.log(f"Saved checkpoint: {ckpt_path}", console=True)

                torch.cuda.synchronize()
                t0_wall = time.time()
                step_t0 = time.perf_counter()

            if last_step:
                break

            # ---------------- training ----------------
            ddp_model.train()

            for i in range(1, grad_accum_steps + 1):
                with autocast_ctx():
                    _, loss = ddp_model(x, y, return_logits=False)
                    train_loss = loss.detach()

                # prefetch next batch
                x, y = train_loader.next_batch()

                if i < grad_accum_steps and hasattr(ddp_model, 'no_sync'):
                    with ddp_model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            # average grads over accumulation
            for p in ddp_model.parameters():
                if p.grad is not None:
                    p.grad.div_(grad_accum_steps)

            # LR update
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            ddp_model.zero_grad(set_to_none=True)

            # ---------------- metrics ----------------
            torch.cuda.synchronize()
            step_dt = time.perf_counter() - step_t0
            step_t0 = time.perf_counter()

            # throughput stats
            cum_tokens += int(tokens_per_step)
            if step >= warm_ignore:
                steps_counted += 1
                sum_step_time += step_dt
            toks_now = tokens_per_step / max(step_dt, 1e-9)
            toks_avg = (
                tokens_per_step * steps_counted / max(sum_step_time, 1e-9)
                if steps_counted > 0
                else float("nan")
            )

            lr = get_lr(step)
            peak = get_peak_mem_mib()
            total_time_s = time.perf_counter() - total_t0

            if master:
                approx_time = training_time_ms + 1000.0 * (time.time() - t0_wall)

                logger.log(
                    f"[train] step {step+1}/{args.num_iterations} "
                    f"(epoch {epoch_idx + 1} + {epoch_frac:.2%}) "
                    f"loss {train_loss.item():.4f} | "
                    f"step_t {step_dt:.3f}s "
                    f"toks/s(now) {toks_now:,.0f} toks/s(avg) {toks_avg:,.0f} | "
                    f"cum_tokens {cum_tokens:,} | "
                    f"lr {lr:.6g} | "
                    f"train_time {approx_time/1000.0:.1f}s total_time {total_time_s:.1f}s | "
                    f"peak_mem {peak} MiB",
                    console=True,
                )

                logger.write_row(
                    dict(
                        ts=now_str(),
                        kind="train",
                        step=step + 1,
                        epoch=float((step + 1) / epoch_steps) if epoch_steps > 0 else 0.0,
                        train_loss=float(train_loss.item()),
                        val_loss=val_loss_scalar,
                        lr=lr,
                        train_time_s=approx_time / 1000.0,
                        total_time_s=total_time_s,
                        step_time_s=step_dt,
                        toks_now=toks_now,
                        toks_avg=toks_avg,
                        cum_tokens=cum_tokens,
                        peak_mem_mib=peak,
                    )
                )
                if use_wandb:
                    wandb.log({
                        "train_loss": float(train_loss.item()),
                        "epoch": float((step + 1) / epoch_steps) if epoch_steps > 0 else 0.0,
                        "lr": lr,
                        "step_time_s": step_dt,
                        "toks_per_sec": toks_now,
                        "toks_per_sec_avg": toks_avg,
                        "cum_tokens": cum_tokens,
                        "peak_mem_mib": peak,
                    }, step=step + 1)

            step += 1

        if master:
            peak = get_peak_mem_mib()
            reason = "target_reached" if target_reached else "max_steps"
            logger.log(f"Done ({reason}). peak memory consumption: {peak} MiB | run_dir={run_dir}", console=True)

    finally:
        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        try:
            logger.close()
        except Exception:
            pass
        barrier()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()