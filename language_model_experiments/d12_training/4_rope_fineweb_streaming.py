#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 with RoPE + AdamW — streaming from HuggingFace FineWeb.

Model: GPT-2 Small with Rotary Position Embeddings (RoPE)
Data: Streams from HuggingFaceFW/fineweb with tiktoken tokenization
Optimizer: AdamW

Same model & training logic as 4_rope_fineweb.py, but streams data instead of using local .bin files.
"""

import os
import sys
import time
import math
import uuid
import argparse
import csv
import json
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
import torch.distributed as dist

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken required. pip install tiktoken", file=sys.stderr)
    sys.exit(1)

print(f"Running pytorch {torch.__version__}")


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


# =============================================================================
# Streaming data loader (HuggingFace + tiktoken)
# =============================================================================

class StreamingDataLoader:
    """
    Streams and tokenizes text from a HuggingFace dataset for GPT-2 training.

    Two modes:
    - Training (cache_mode=False): Continuously streams, tokenizes, and buffers tokens.
    - Validation (cache_mode=True): Streams and caches a fixed number of tokens at init.
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

        # Pinned CPU buffer
        self._cpu_buf_u16 = torch.empty((B * T + 1,), dtype=torch.uint16, pin_memory=True)

        # Token buffer
        self.tokens: np.ndarray = np.empty(0, dtype=np.uint16)
        self.current_position = 0
        self._stream = None

        # Source info
        self.source_name = f"{hf_dataset}/{hf_config}"

        # Start streaming
        self._stream = self._create_stream()

        if rank == 0:
            mode_str = "val-cache" if cache_mode else "train-stream"
            print(f"[data:{mode_str}] Streaming from {self.source_name} "
                  f"(split={hf_split}, shuffle={shuffle})...", flush=True)

        self._fill_buffer(buffer_tokens)

        if rank == 0:
            print(f"[data:{mode_str}] Initial buffer: {len(self.tokens):,} tokens", flush=True)

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
                    break
                self._shuffle_seed += 1
                self._stream = self._create_stream()

        if not chunks:
            return

        new_data = np.concatenate(chunks)

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
        """Reset position to start."""
        self.current_position = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, W = self.B, self.T, self.world_size
        global_bt = W * B * T
        needed = global_bt + 1
        available = len(self.tokens) - self.current_position

        if available < needed:
            if self.cache_mode:
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

        x = self._cpu_buf_u16[:-1].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        y = self._cpu_buf_u16[1:].view(B, T).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y


# =============================================================================
# Model: GPT-2 with RoPE (identical to 4_rope_fineweb.py)
# =============================================================================

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
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(dtype=torch.bfloat16)
            self.sin_cached = freqs.sin().to(dtype=torch.bfloat16)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
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

        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer


def save_training_checkpoint(
    step: int,
    raw_model: nn.Module,
    optimizers,
    output_dir: str,
    val_loss: Optional[float],
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
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


def get_args():
    p = argparse.ArgumentParser("GPT-2 RoPE Training with Streaming Data")

    # HuggingFace streaming
    p.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--hf_config", type=str, default="sample-100BT")
    p.add_argument("--hf_val_config", type=str, default=None)
    p.add_argument("--hf_split", type=str, default="train")
    p.add_argument("--buffer_tokens", type=int, default=20_000_000)
    p.add_argument("--epoch_tokens", type=int, default=100_000_000_000)

    # Output
    p.add_argument("--output_dir", type=str, required=True)

    # Model
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=768)

    # Batching
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device_batch_size", type=int, default=64)
    p.add_argument("--sequence_length", type=int, default=1024)
    p.add_argument("--total_batch_size", type=int, default=None)

    # Optimization
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_iterations", type=int, default=50000)
    p.add_argument("--warmup_iters", type=int, default=0)
    p.add_argument("--learning_rate_decay_frac", type=float, default=0.0)
    p.add_argument("--zero_stage", type=int, default=0)

    # Validation cadence
    p.add_argument("--val_every_steps", type=int, default=100)
    p.add_argument("--val_every_epoch_frac", type=float, default=0.0)
    p.add_argument("--val_tokens", type=int, default=10485760)

    # Checkpoint cadence
    p.add_argument("--checkpoint_every_steps", type=int, default=800)
    p.add_argument("--save_every_epoch_frac", type=float, default=0.0)

    # Stopping criteria
    p.add_argument("--stop_mode", type=str, default="const_loss", choices=["const_loss", "epoch"])
    p.add_argument("--loss_threshold", type=float, default=3.2)
    p.add_argument("--stop_epoch_frac", type=float, default=None)

    # System
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--tensorcores", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


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
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = device_type
        if args.device:
            device = args.device

    set_seed(args.seed, ddp_rank)

    # Logging
    logfile = None
    csv_file = None
    csv_writer = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        with open(logfile, "w") as f:
            f.write(f"Log started at {time.asctime()}\n")

        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        csv_path = os.path.join(args.output_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "timestamp", "step", "train_loss", "val_loss",
            "lr", "grad_norm", "step_time_ms", "tokens_per_sec",
            "train_tokens", "total_tokens", "peak_mem_mib"
        ])
        csv_writer.writeheader()

    def log_print(*a, **k):
        if not master_process:
            return
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
    from contextlib import nullcontext
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if args.tensorcores and device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Batch sizes
    B = args.device_batch_size
    T = args.sequence_length
    global_batch_size_seq = args.batch_size
    assert global_batch_size_seq % (B * ddp_world_size) == 0
    grad_accum_steps = global_batch_size_seq // (B * ddp_world_size)
    tokens_per_step = global_batch_size_seq * T

    if master_process:
        log_print(f"Batch config: Global B={global_batch_size_seq}, Device B={B}, Grad Accum={grad_accum_steps}, T={T}")
        log_print(f"Tokens per step: {tokens_per_step:,}")

    # Data Loaders (streaming)
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
        device=str(device),
    )

    train_loader = StreamingDataLoader(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        B=B, T=T, rank=ddp_rank, world_size=ddp_world_size,
        buffer_tokens=args.buffer_tokens,
        cache_mode=False,
        shuffle=True,
        shuffle_seed=args.seed,
        device=str(device),
        epoch_tokens=args.epoch_tokens,
    )

    # Val steps
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_max_steps = args.val_tokens // (B * T * ddp_world_size)

    # Epoch calculation
    total_train_tokens = args.epoch_tokens
    epoch_steps = max(1, total_train_tokens // tokens_per_step)

    if master_process:
        log_print(f"Train source: {train_loader.source_name} | epoch_tokens={total_train_tokens:,}")
        log_print(f"Val source: {val_loader.source_name} | cached_tokens={val_loader.ntok_total:,}")
        log_print(f"Epoch: {epoch_steps:,} steps per epoch, {args.num_iterations / epoch_steps:.2f} total epochs")

    # Validation cadence
    if args.val_every_steps > 0:
        val_loss_every = args.val_every_steps
    elif args.val_every_epoch_frac > 0:
        val_loss_every = max(1, int(round(args.val_every_epoch_frac * epoch_steps)))
    else:
        val_loss_every = 0

    # Checkpoint cadence
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

    # LR Schedule
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

    train_loader.reset()
    x, y = train_loader.next_batch()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # Save init checkpoint
    if master_process:
        p0 = save_training_checkpoint(0, raw_model, optimizer, args.output_dir, val_loss=None)
        log_print(f"💾 Saved initial checkpoint: {p0}")

    free_mib, total_mib = get_free_total_mib()
    log_print(f"\nGPU Memory: peak {get_peak_mem_mib()} MiB | free/total {free_mib}/{total_mib} MiB")
    log_print(f"{'='*60}\nTRAINING STARTED\n{'='*60}")

    if device_type == "cuda":
        torch.cuda.synchronize()

    wall_t0 = time.time()

    while step < args.num_iterations:
        last_step = (step == args.num_iterations - 1)

        if step == skip_warmup_steps:
            training_time_s = 0.0
            torch.cuda.synchronize()
            wall_t0 = time.time()

        epoch_idx = step // epoch_steps
        step_in_epoch = step % epoch_steps
        epoch_frac = (step_in_epoch / epoch_steps) if epoch_steps > 0 else 0.0

        # Validation
        if val_loss_every > 0 and (step % val_loss_every == 0 or last_step):
            torch.cuda.synchronize() if device_type == "cuda" else None

            model.eval()
            val_loader.reset()
            if device_type == "cuda":
                torch.cuda.synchronize()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(val_max_steps):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
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

                curr_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - wall_t0

                avg_tps = (train_tokens / training_time_s) if training_time_s > 0 else 0.0
                peak = get_peak_mem_mib()
                reserved = get_reserved_mem_mib()
                free, total = get_free_total_mib()

                log_print(f"\n{'─'*70}")
                log_print(f"[VAL] step {step:,}/{args.num_iterations:,} | epoch {epoch_idx + 1} + {epoch_frac:.2%}")
                log_print(f"      val_loss {val_loss:.6f} | threshold {args.loss_threshold}")
                log_print(f"      lr {curr_lr:.4g}")
                log_print(f"      train_time {training_time_s:.1f}s | wall {elapsed:.1f}s")
                log_print(f"      tokens train {train_tokens:,} | val {val_tokens_count:,} | total {total_tokens_now:,}")
                log_print(f"      avg throughput {avg_tps:,.0f} tok/s")
                log_print(f"      mem peak {peak} MiB | reserved {reserved} MiB | free {free}/{total} MiB")
                log_print(f"{'─'*70}\n")

                # Stopping criteria
                stop_now = False
                if args.stop_mode == "const_loss":
                    if (args.loss_threshold > 0) and (val_loss < args.loss_threshold) and (step > 0):
                        log_print(f"🎯 EARLY STOP: val_loss {val_loss:.4f} < {args.loss_threshold}")
                        stop_now = True
                elif args.stop_mode == "epoch":
                    current_epoch_frac = step / epoch_steps
                    if (args.stop_epoch_frac is not None) and (current_epoch_frac >= args.stop_epoch_frac):
                        log_print(f"🎯 EPOCH STOP: Reached {current_epoch_frac:.3f} epochs (target: {args.stop_epoch_frac})")
                        stop_now = True

                if stop_now:
                    save_training_checkpoint(step, raw_model, optimizer, args.output_dir, val_loss)
                    break

            model.train()
            wall_t0 = time.time()

        # Training Step
        optimizer.zero_grad(set_to_none=True)

        step_loss_acc = torch.tensor(0.0, device=device)

        t0 = time.perf_counter()

        for micro_step in range(grad_accum_steps):
            is_last = (micro_step == grad_accum_steps - 1)
            if ddp:
                model.require_backward_grad_sync = is_last

            with ctx:
                _, loss = model(x, y, return_logits=False)
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
            if gn is not None:
                grad_norm = float(gn)

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

        # Logging
        if master_process:
            tps = tokens_per_step / max(dt, 1e-9)
            step_loss = step_loss_acc.item()

            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"loss {step_loss:.4f} │ "
                f"gnorm {grad_norm:6.2f} │ "
                f"lr {lr:.6f} │ "
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

            # Checkpoint
            if checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
                save_training_checkpoint(step + 1, raw_model, optimizer, args.output_dir, None)

        step += 1

    # End
    if master_process:
        log_print("Training Finished.")
        if csv_file:
            csv_file.close()
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
