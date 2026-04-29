#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 small baseline with AdamW + DDP — streaming from HuggingFace FineWeb.

Same model & training logic as 3_baseline.py, but instead of reading
pre-tokenized .bin shards from disk, this script:
  - Streams text from HuggingFaceFW/fineweb (sample-100BT by default)
  - Tokenizes on-the-fly with tiktoken (GPT-2 encoding)
  - Buffers tokens in memory for efficient batching
  - Caches a fixed set of validation tokens at startup for deterministic eval

Requires:
  pip install tiktoken datasets

Example (single GPU):
  srun python 3_baseline_streaming.py \\
    --hf_dataset "HuggingFaceFW/fineweb" \\
    --hf_config "sample-100BT" \\
    --out_dir "/path/runs/gpt2_stream" \\
    --run_name "gpt2_stream_baseline" \\
    --num_iterations 60000 --compile --bf16

Note: Compute nodes must have internet access for HuggingFace streaming.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import argparse
import csv
import json
import math
import signal
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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

# Required: tiktoken
try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken is required. Install with: pip install tiktoken", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# Small utilities
# =============================================================================

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


# =============================================================================
# Model (identical to 3_baseline.py)
# =============================================================================

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
    p = argparse.ArgumentParser("gpt2_baseline_streaming")

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

    # paths
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
        logger.log(f"Streaming dataset: {args.hf_dataset}/{args.hf_config} (split={args.hf_split})", console=True)
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
    # data (streaming from HuggingFace)
    # -------------------------------------------------------------------------
    # Validation: cache a fixed set of tokens at startup (deterministic, no shuffle)
    val_cache_tokens = args.val_tokens + world_size * B * T + 1
    val_config = args.hf_val_config or args.hf_config

    val_loader = StreamingDataLoader(
        hf_dataset=args.hf_dataset,
        hf_config=val_config,
        hf_split=args.hf_split,
        B=B, T=T, rank=rank, world_size=world_size,
        buffer_tokens=val_cache_tokens,
        cache_mode=True,
        shuffle=False,
        device=str(device),
    )

    # Training: continuous streaming with shuffle
    train_loader = StreamingDataLoader(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        B=B, T=T, rank=rank, world_size=world_size,
        buffer_tokens=args.buffer_tokens,
        cache_mode=False,
        shuffle=True,
        shuffle_seed=args.seed,
        device=str(device),
        epoch_tokens=args.epoch_tokens,
    )

    # ---------------- epoch definition ----------------
    epoch_tokens = int(args.epoch_tokens)
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
        logger.log(f"Train source: {train_loader.source_name} | epoch_tokens={epoch_tokens:,}", console=True)
        logger.log(f"Val source:   {val_loader.source_name} | cached_tokens={val_loader.ntok_total:,}", console=True)
        logger.log(f"grad_accum_steps={grad_accum_steps} | per-rank batch={B} | seq_len={T}", console=True)
        logger.log(f"tokens_per_step={tokens_per_step:,}", console=True)
        logger.log(f"epoch_tokens(nominal)={epoch_tokens:,} | epoch_steps={epoch_steps:,} | epoch_tokens(effective)={epoch_tokens_effective:,}", console=True)
        logger.log(f"Epoch: {epoch_steps:,} steps per epoch, {args.num_iterations / epoch_steps:.4f} total epochs", console=True)
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
    # LR schedule: cosine decay with warmup
    # -------------------------------------------------------------------------
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
