# =================================================================
# GPT-2 Training with Value Embeddings (VE/VTE) + FlexAttention
# =================================================================

import os, sys, math, glob, time, argparse, csv, json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple
from scale_arch_utils import (
    build_sparse_endpoint_pattern,
    default_skip_attn_layer,
    default_vte_endpoint_k,
    resolve_model_dims,
    validate_dims,
)
from scale_dist_utils import (
    all_reduce_mean_inplace,
    destroy_distributed,
    init_distributed,
    resolve_compile_enabled,
    resolve_loss_fp32_enabled,
    wrap_model_fsdp,
)

import numpy as np
import wandb
import torch
import sys

HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch._inductor.config as inductor_config

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    flex_attention = torch.compile(flex_attention, dynamic=False)
except Exception as e:
    if not HELP_REQUESTED:
        raise RuntimeError("This script requires torch.nn.attention.flex_attention (flex_attention + BlockMask).") from e
    flex_attention = None
    BlockMask = None

import torch._dynamo as dynamo
dynamo.reset()

print(f"Running PyTorch {torch.__version__}")

def is_master() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def print0(*args, **kwargs):
    if is_master():
        print(*args, **kwargs)

def print_gpu_inventory():
    if not torch.cuda.is_available():
        print0("CUDA not available")
        return
    n = torch.cuda.device_count()
    total_mem_gb = 0.0
    names = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        cc = f"{props.major}.{props.minor}"
        print0(f"GPU {i}: {props.name} | {mem_gb:.1f} GB | CC {cc}")
        names.append(props.name)
        total_mem_gb += mem_gb
    h200 = sum("H200" in nm.upper() for nm in names)
    print0(f"\nDetected {n} CUDA device(s). H200 guess: {h200}. Total VRAM: {total_mem_gb:.1f} GB.")

print_gpu_inventory()

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def zeropower_via_svd(G: Tensor, steps=None) -> Tensor:
    # SVD is slow but stable; mostly for debugging.
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to approximate UV^T (orthogonalization).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    if transposed:
        X = X.T
    return X

ZEROP_POWER = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5,
}

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend = ZEROP_POWER[group["backend"]]
            steps = group["backend_steps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                gg = g
                if group["nesterov"]:
                    gg = gg.add(buf, alpha=momentum)

                gg = backend(gg, steps=steps)
                gg = gg * (max(1.0, gg.size(0) / gg.size(1)) ** 0.5)

                p.add_(gg.to(dtype=p.dtype), alpha=-lr)

def apply_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: Tensor):
        # x: [B, T, H, D]
        seq_len = x.shape[1]
        key = (x.device, x.dtype, seq_len)
        if key != self._cache_key:
            self._cache_key = key
            inv = self.inv_freq.to(device=x.device)
            t = torch.arange(seq_len, device=x.device, dtype=inv.dtype)
            freqs = torch.outer(t, inv)
            self.cos_cached = freqs.cos().to(dtype=x.dtype)
            self.sin_cached = freqs.sin().to(dtype=x.dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, T, H, D]
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))  # [base_v, ve_v]
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: Tensor, ve: Optional[Tensor], block_mask: BlockMask) -> Tensor:
        # x: [1, T, dim], ve: [T, dim] or [1,T,dim] or None
        B, T, _ = x.shape
        assert B == 1, "FlexAttention setup assumes batch size 1"

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)

        if ve is None:
            v = self.mix[0] * v
        else:
            if ve.ndim == 2:
                ve = ve[None]  # -> [1, T, dim]
            assert ve.shape == (B, T, self.dim), f"Expected ve {(B,T,self.dim)}, got {tuple(ve.shape)}"
            ve = ve.view(B, T, self.num_heads, self.head_dim)
            v = self.mix[0] * v + self.mix[1] * ve.to(dtype=v.dtype)

        q = apply_norm(q)
        k = apply_norm(k)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc   = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, layer_idx: int, skip_attn_layer: Optional[int] = 7):
        super().__init__()
        self.layer_idx = layer_idx
        self.skip_attn_layer = skip_attn_layer
        self.attn = CausalSelfAttention(dim, n_head) if layer_idx != skip_attn_layer else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))  # mix current x and x0

    def forward(self, x: Tensor, ve: Optional[Tensor], x0: Tensor, block_mask: BlockMask) -> Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(apply_norm(x), ve, block_mask)
        x = x + self.mlp(apply_norm(x))
        return x

class ValueTokenEmbedding(nn.Module):
    """
    Your "sparse embedding" scheme: 3 embedding tables and a fixed per-layer pattern.
    """
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, endpoint_k: int = 0):
        super().__init__()
        self.n_layer = n_layer
        self.endpoint_k = endpoint_k
        self.emb = nn.ModuleList([nn.Embedding(vocab_size, n_embd) for _ in range(3)])

    def forward(self, tokens_1d: Tensor):
        # tokens_1d: [T]
        ve0 = self.emb[0](tokens_1d)  # [T, n_embd]
        ve1 = self.emb[1](tokens_1d)
        ve2 = self.emb[2](tokens_1d)

        ve = build_sparse_endpoint_pattern(ve0, ve1, ve2, self.n_layer, self.endpoint_k)
        assert len(ve) == self.n_layer, f"VE pattern length {len(ve)} != n_layer {self.n_layer}"
        return ve

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    skip_attn_layer: int = -1
    vte_endpoint_k: int = 0

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, lm_head_softcap: float = 30.0):
        super().__init__()
        self.config = config
        self.lm_head_softcap = float(lm_head_softcap)

        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h   = nn.ModuleList([Block(config.n_embd, config.n_head, i, skip_attn_layer=config.skip_attn_layer)
                                for i in range(config.n_layer)]),
        ))
        self.value_embeds = ValueTokenEmbedding(
            config.vocab_size,
            config.n_embd,
            config.n_layer,
            endpoint_k=config.vte_endpoint_k,
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def _make_block_mask(self, tokens_1d: Tensor, sliding_window_num_blocks: Tensor, block_size: int = 128) -> BlockMask:
        # Assumption: T divisible by block_size (true for 65536 and block_size=128)
        T = tokens_1d.numel()
        assert T % block_size == 0, f"T={T} must be divisible by BLOCK_SIZE={block_size}"
        n_blocks = T // block_size

        # Document ids per token
        docs = (tokens_1d == 50256).cumsum(0)

        # Block-level doc range summaries
        docs_low  = docs.view(n_blocks, block_size)[:, 0].contiguous()
        docs_high = docs.view(n_blocks, block_size)[:, -1].contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            # q_idx/kv_idx are token indices
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

        sw = torch.as_tensor(sliding_window_num_blocks, dtype=torch.int32, device=tokens_1d.device).clamp_min(1)
        kv_idx = torch.arange(n_blocks, dtype=torch.int32, device=tokens_1d.device)
        q_idx  = kv_idx[:, None]

        causal_bm  = q_idx >= kv_idx
        window_bm  = (q_idx - kv_idx) < sw
        document_bm = (docs_low[:, None] <= docs_high[None, :]) & (docs_low[None, :] <= docs_high[:, None])

        dense = causal_bm & window_bm & document_bm
        nb = dense.sum(dim=-1, dtype=torch.int32)
        idx = dense.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
        nb  = nb[None, None].contiguous()
        idx = idx[None, None].contiguous()

        return BlockMask.from_kv_blocks(nb, idx, BLOCK_SIZE=block_size, mask_mod=mask_mod)

    def forward(self, inputs: Tensor, targets: Tensor, sliding_window_num_blocks: Tensor) -> Tensor:
        # inputs/targets: [B,T] or [T], B must be 1 for flex attention
        if inputs.ndim == 2:
            assert inputs.size(0) == 1
            inputs = inputs[0]
        if targets.ndim == 2:
            assert targets.size(0) == 1
            targets = targets[0]
        assert inputs.ndim == 1 and targets.ndim == 1
        assert inputs.numel() == targets.numel()

        block_mask = self._make_block_mask(inputs, sliding_window_num_blocks, block_size=128)

        x = self.transformer.wte(inputs[None])   # [1,T,C]
        x = apply_norm(x)
        x0 = x

        ve_all = self.value_embeds(inputs)  # list length n_layer of [T,C] or None

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_all[i], x0, block_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, ve_all[self.num_encoder_layers + i], x0, block_mask)

        x = apply_norm(x)
        logits = self.lm_head(x)
        sc = self.lm_head_softcap
        logits = sc * torch.tanh(logits / sc)
        if getattr(self, "loss_fp32", True):
            logits = logits.float()

        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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

def save_training_checkpoint(
    step: int,
    raw_model: nn.Module,
    optimizers,
    output_dir: str,
    val_loss: Optional[float],
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "step": int(step),
        "val_loss": None if val_loss is None else float(val_loss),
        "model": raw_model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "rng": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "extra": extra or {},
    }
    path = os.path.join(output_dir, f"ckpt_step_{step:07d}.pt")
    torch.save(ckpt, path)
    return path

def get_args():
    p = argparse.ArgumentParser("GPT-2 Training with VE/VTE + FlexAttention")

    # Data
    p.add_argument("--train_pattern", type=str, required=True, help="Glob pattern for training data shards")
    p.add_argument("--val_pattern", type=str, default="", help="Glob pattern for validation data shards")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs")

    # Model
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--model_profile", type=str, default="d12", choices=["d12", "d24", "d36", "d48"])
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--skip_attn_layer", type=int, default=-1)
    p.add_argument("--vte_endpoint_k", type=int, default=0)
    p.add_argument("--lm_head_softcap", type=float, default=30.0)

    # Batching
    p.add_argument("--batch_size", type=int, default=8)         # global sequences
    p.add_argument("--device_batch_size", type=int, default=1)  # must be 1 for flex attention
    p.add_argument("--sequence_length", type=int, default=65536)
    p.add_argument("--val_tokens", type=int, default=10485760, help="Total validation tokens to use")

    # LRs
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
    p.add_argument("--loss_threshold", type=float, default=3.3, help="[const_loss mode] Early stop when val loss < threshold")
    p.add_argument("--stop_epoch_frac", type=float, default=None,
                   help="[epoch mode] Stop at this epoch fraction (e.g., 0.5). Required if stop_mode='epoch'.")

    # Window schedule (TOKEN units here; we convert to NUM_BLOCKS)
    p.add_argument("--window_min", type=int, default=64)
    p.add_argument("--window_max", type=int, default=1792)
    p.add_argument("--window_warmup_steps", type=int, default=3000)
    p.add_argument("--block_size", type=int, default=128)    # Validation cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--val_every_steps", type=int, default=100,
                   help="Validate every N steps (0 to disable). If >0, takes precedence over --val_every_epoch_frac.")
    p.add_argument("--val_every_epoch_frac", type=float, default=0.0,
                   help="Validate every fraction of an epoch (e.g., 0.025 => 1/40 epoch). Ignored if --val_every_steps > 0.")    # Checkpoint cadence (mutually exclusive: step-based OR epoch-fraction-based)
    p.add_argument("--checkpoint_every_steps", type=int, default=400,
                   help="Save checkpoint every N steps (0 to disable). If >0, takes precedence over --save_every_epoch_frac.")
    p.add_argument("--save_every_epoch_frac", type=float, default=0.0,
                   help="Save checkpoint every fraction of an epoch. Ignored if --checkpoint_every_steps > 0.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--parallel_mode", type=str, default="auto", choices=["auto", "single", "ddp", "fsdp"])
    p.add_argument("--compile_mode", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--activation_checkpointing", type=str, default="auto", choices=["auto", "none", "block"])
    p.add_argument("--loss_fp32", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--tensorcores", action="store_true")
    p.add_argument("--use_cudnn_attn", action="store_true")

    # WandB
    p.add_argument("--wandb_project", type=str, default="gpt2-dynamics")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="13_sparse_embd_gpt2")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=1)

    args = p.parse_args()
    args.n_layer, args.n_head, args.n_embd = resolve_model_dims(args, default=(12, 6, 768))
    validate_dims(args.n_layer, args.n_head, args.n_embd, require_even_layers=True)
    if args.skip_attn_layer < 0:
        args.skip_attn_layer = default_skip_attn_layer(args.n_layer)
    if args.vte_endpoint_k <= 0:
        args.vte_endpoint_k = default_vte_endpoint_k(args.n_layer)
    if getattr(args, "compile", False):
        args.compile_mode = "on"
    args.compile_enabled = resolve_compile_enabled(args.compile_mode, args.model_profile, int(os.environ.get("WORLD_SIZE", "1")))
    args.loss_fp32_enabled = resolve_loss_fp32_enabled(args.loss_fp32, args.model_profile)
    if args.activation_checkpointing == "auto":
        args.activation_checkpointing = "block" if args.model_profile in {"d36", "d48"} else "none"
    return args

def main():
    args = get_args()
    dist_env = init_distributed(args.parallel_mode, fsdp_default=True)
    ddp = dist_env.mode == "ddp"
    fsdp = dist_env.mode == "fsdp"
    ddp_rank = dist_env.rank
    ddp_local_rank = dist_env.local_rank
    ddp_world_size = dist_env.world_size
    master_process = dist_env.master
    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed, ddp_rank)

    def _sync():
        if device_type == "cuda":
            torch.cuda.synchronize()

    # dtype / autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # (Optional) CUDNN attention (usually irrelevant for flex_attention, but keep your toggle)
    if args.use_cudnn_attn and device_type == "cuda":
        try:
            from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
            enable_cudnn_sdp(True)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            if master_process: print("Enabled CUDNN SDP")
        except Exception as e:
            if master_process: print(f"Warning enabling CUDNN SDP: {e}")

    # Compile flex_attention only when NOT compiling whole model (avoid nested dynamo weirdness)
    global flex_attention
    if not args.compile_enabled:
        flex_attention = torch.compile(flex_attention, dynamic=False)

    # Model
    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        skip_attn_layer=args.skip_attn_layer,
        vte_endpoint_k=args.vte_endpoint_k,
    )
    model = GPT(config, lm_head_softcap=args.lm_head_softcap).to(device)
    model.train()
    model.loss_fp32 = args.loss_fp32_enabled
    # Cache optimizer parameter references before distributed wrapping can flatten block params.
    opt_ve_params = list(model.value_embeds.parameters())
    opt_block_params = list(model.transformer.h.parameters())
    opt_wte_weight = model.transformer.wte.weight
    opt_lm_head_weight = model.lm_head.weight
    opt_skip_weights = model.skip_weights
    opt_matrix_params = [p for p in opt_block_params if p.ndim == 2]
    opt_scalar_params = [p for p in opt_block_params if p.ndim < 2]

    if args.compile_enabled:
        if master_process:
            print("Compiling model with torch.compile() ...")
        if inductor_config is not None:
            inductor_config.coordinate_descent_tuning = True
        model = torch.compile(model)

    # Data
    B, T = args.device_batch_size, args.sequence_length
    assert B == 1, "FlexAttention path assumes device_batch_size=1"
    assert T % args.block_size == 0, f"sequence_length {T} must be divisible by block_size {args.block_size}"

    train_loader = DistributedDataLoader(args.train_pattern, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.val_pattern, B, T, ddp_rank, ddp_world_size) if args.val_pattern else None

    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.batch_size % (B * ddp_world_size) == 0
    grad_accum_steps = args.batch_size // (B * ddp_world_size)
    total_batch_tokens = tokens_per_fwdbwd * grad_accum_steps
    epoch_steps = max(1, int(math.ceil(train_loader.ntok_total / max(1, total_batch_tokens))))

    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_max_steps = args.val_tokens // (B * T * ddp_world_size)

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

    if fsdp:
        model = wrap_model_fsdp(model, dist_env, block_types=(Block,), activation_checkpointing=args.activation_checkpointing)
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
    elif ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if (ddp or fsdp) else model

    # Optimizers
    matrix_params = list(opt_matrix_params)
    scalar_params = list(opt_scalar_params) + [opt_skip_weights]

    opt_embed = torch.optim.Adam(
        opt_ve_params + [opt_wte_weight],
        lr=args.embed_lr, betas=(0.8, 0.95), fused=(device_type == "cuda")
    )
    opt_head = torch.optim.Adam(
        [opt_lm_head_weight],
        lr=args.head_lr, betas=(0.9, 0.95), fused=(device_type == "cuda")
    )
    opt_muon = Muon(matrix_params, lr=args.muon_lr, momentum=args.muon_momentum_init)
    opt_scalar = torch.optim.Adam(
        scalar_params,
        lr=args.scalar_lr, betas=(0.8, 0.95), fused=(device_type == "cuda")
    )

    optimizers = [opt_embed, opt_head, opt_muon, opt_scalar]

    # LR schedule
    total_steps = args.num_iterations
    warmup_steps = int(round(args.warmup_frac * total_steps))
    warmdown_steps = int(round(args.warmdown_frac * total_steps))
    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)

    def lr_mult(it: int) -> float:
        it = min(it, total_steps)
        if warmup_steps > 0 and it < warmup_steps:
            return (it + 1) / warmup_steps
        decay_start = warmup_steps + plateau_steps
        if warmdown_steps <= 0 or it < decay_start:
            return 1.0
        return max(0.0, (total_steps - it) / max(1, warmdown_steps))

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_mult) for opt in optimizers]

    # WandB
    wandb_run = None
    if master_process and args.wandb_mode != "disabled":
        os.makedirs(args.output_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", args.output_dir)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=["ve", "flex_attention", "unet"],
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb.define_metric("global_step")
        for ns in ["train/*", "val/*", "loss/*", "lr/*", "perf/*", "mem/*"]:
            wandb.define_metric(ns, step_metric="global_step")

    # Logging files
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

    def log_print(*a, **k):
        if not master_process:
            return
        msg = " ".join(str(x) for x in a)
        print(msg, **k)
        if logfile:
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    def log_row(row: dict):
        if csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()

    # Save step-0
    if master_process:
        p0 = save_training_checkpoint(0, raw_model, optimizers, args.output_dir, val_loss=None)
        log_print(f"💾 Saved initial checkpoint: {p0}")

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Training state
    training_time_s = 0.0
    train_tokens = 0
    val_tokens_count = 0
    skip_warmup_steps = 10
    wall_t0 = time.time()
    step = 0
    val_loss = float("nan")
    saved_final = False
    stop_now = False

    # Prefetch first batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # Sliding window tensor (NUM BLOCKS)
    sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device=device)

    free_mib, total_mib = get_free_total_mib()
    log_print(f"\nGPU Memory: peak {get_peak_mem_mib()} MiB | free/total {free_mib}/{total_mib} MiB")
    log_print(f"{'='*60}\nTRAINING STARTED\n{'='*60}")

    while True:
        last_step = (step == args.num_iterations)

        # --- Window schedule (tokens -> blocks)
        wfrac = min(step / max(1, args.window_warmup_steps), 1.0)
        window_tokens = int(args.window_min + wfrac * (args.window_max - args.window_min))
        # round down to multiple of block_size, then convert to num blocks (>=1)
        window_tokens = max(args.block_size, (window_tokens // args.block_size) * args.block_size)
        sw_blocks = max(1, window_tokens // args.block_size)
        sliding_window_num_blocks.fill_(sw_blocks)

        # --- Muon momentum warmup
        mfrac = min(step / max(1, args.muon_momentum_warmup_steps), 1.0)
        muon_momentum = (1 - mfrac) * args.muon_momentum_init + mfrac * args.muon_momentum_final

        run_val = (val_loader is not None) and (val_loss_every > 0) and (step % val_loss_every == 0 or last_step)
        if run_val:
            _sync()
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_t = torch.tensor(0.0, device=device)
                for _ in range(val_max_steps):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
                    with ctx:
                        lv = model(xv, yv, sliding_window_num_blocks)
                    val_loss_t += lv.detach()
                val_loss_t /= max(1, val_max_steps)
                if dist_env.distributed:
                    all_reduce_mean_inplace(val_loss_t, dist_env)
                val_loss = float(val_loss_t.item())

            val_batch_tokens = val_max_steps * B * T * ddp_world_size
            val_tokens_count += val_batch_tokens
            total_tokens = train_tokens + val_tokens_count

            # Epoch-ish info
            tokens_per_epoch = train_loader.ntok_total
            epoch = (train_tokens / tokens_per_epoch) if tokens_per_epoch > 0 else 0.0
            epoch_pct = (epoch % 1.0) * 100.0

            avg_tps = (train_tokens / training_time_s) if training_time_s > 0 else 0.0
            peak = get_peak_mem_mib()
            reserved = get_reserved_mem_mib()
            free, total = get_free_total_mib()

            log_print(f"\n{'─'*70}")
            log_print(f"[VAL] step {step:,}/{args.num_iterations:,} | epoch {int(epoch)+1} ({epoch_pct:.1f}%)")
            log_print(f"      window {window_tokens} tok = {sw_blocks} blocks (min {args.window_min} → max {args.window_max})")
            log_print(f"      val_loss {val_loss:.6f} | threshold {args.loss_threshold}")
            log_print(f"      lr embed={opt_embed.param_groups[0]['lr']:.4g} head={opt_head.param_groups[0]['lr']:.4g} "
                      f"muon={opt_muon.param_groups[0]['lr']:.4g} scalar={opt_scalar.param_groups[0]['lr']:.4g}")
            log_print(f"      μ {muon_momentum:.4f} | train_time {training_time_s:.1f}s | wall {time.time()-wall_t0:.1f}s")
            log_print(f"      tokens train {train_tokens:,} | val {val_tokens_count:,} | total {total_tokens:,}")
            log_print(f"      avg throughput {avg_tps:,.0f} tok/s")
            log_print(f"      mem peak {peak} MiB | reserved {reserved} MiB | free {free}/{total} MiB")
            log_print(f"{'─'*70}\n")

            if master_process and wandb_run is not None:
                wandb.log({
                    "global_step": step,
                    "loss/val": val_loss,
                    "val/loss": val_loss,
                    "val/window_tokens": window_tokens,
                    "val/window_blocks": sw_blocks,
                    "val/epoch": epoch,
                })

            # Stopping criteria - mutually exclusive modes
            epoch_frac = epoch
            if args.stop_mode == "const_loss":
                # Mode 1: Stop by constant loss threshold
                if (val_loss < args.loss_threshold) and (step > 0):
                    log_print(f"🎯 [EARLY STOP - const_loss mode] val_loss {val_loss:.4f} < threshold {args.loss_threshold:.4f}")
                    if master_process:
                        p = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        log_print(f"💾 Saved: {p}")
                        saved_final = True
                    stop_now = True
            elif args.stop_mode == "epoch":
                # Mode 2: Stop by epoch fraction
                if (args.stop_epoch_frac is not None) and (epoch_frac >= args.stop_epoch_frac) and (step > 0):
                    log_print(f"🎯 [EARLY STOP - epoch mode] Epoch fraction {epoch_frac:.4f} >= threshold {args.stop_epoch_frac:.4f}")
                    if master_process:
                        p = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                        log_print(f"💾 Saved: {p}")
                        saved_final = True
                    stop_now = True

            if last_step and master_process and not saved_final:
                p = save_training_checkpoint(step, raw_model, optimizers, args.output_dir, val_loss)
                log_print(f"💾 Final checkpoint: {p}")
                saved_final = True

            model.train()

        if last_step or stop_now:
            break

        # ---- Train step
        _sync()
        t0 = time.perf_counter()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        step_losses = []
        for gas in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (gas == grad_accum_steps - 1)
            with ctx:
                loss = model(x, y, sliding_window_num_blocks)
                (loss / grad_accum_steps).backward()
                step_losses.append(loss.detach())
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Clip
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            gnorm = float(grad_norm) if grad_norm is not None else -1.0
        else:
            gnorm = -1.0

        # set muon momentum
        opt_muon.param_groups[0]["momentum"] = muon_momentum

        for opt, sch in zip(optimizers, schedulers):
            opt.step()
            sch.step()

        _sync()
        dt = time.perf_counter() - t0

        train_tokens += total_batch_tokens
        if step >= skip_warmup_steps:
            training_time_s += dt

        step_loss = torch.stack(step_losses).mean() if step_losses else torch.tensor(0.0, device=device)
        if dist_env.distributed:
            all_reduce_mean_inplace(step_loss, dist_env)
        train_loss = float(step_loss.item())

        # Logging (train line)
        if master_process:
            lr_e = opt_embed.param_groups[0]["lr"]
            lr_h = opt_head.param_groups[0]["lr"]
            lr_m = opt_muon.param_groups[0]["lr"]
            lr_s = opt_scalar.param_groups[0]["lr"]
            tps = total_batch_tokens / max(dt, 1e-9)
            total_tokens_now = train_tokens + val_tokens_count

            log_str = (
                f"step {step+1:5d}/{args.num_iterations} │ "
                f"win {window_tokens:4d} │ "
                f"loss {train_loss:.4f} │ "
                f"gnorm {gnorm:6.2f} │ "
                f"lr {lr_e:.3g}/{lr_h:.3g}/{lr_m:.3g}/{lr_s:.3g} │ "
                f"μ {muon_momentum:.3f} │ "
                f"{dt*1000:7.1f}ms │ "
                f"{tps/1000:7.1f}k tok/s │ "
                f"tokens {total_tokens_now/1e6:7.1f}M"
            )
            if device_type == "cuda":
                log_str += f" │ mem {get_peak_mem_mib()}MiB"
            log_print(log_str)

            log_row({
                "timestamp": now_ts(),
                "kind": "train",
                "step": step + 1,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": "",
                "lr_embed": f"{lr_e:.6g}",
                "lr_head": f"{lr_h:.6g}",
                "lr_muon": f"{lr_m:.6g}",
                "lr_scalar": f"{lr_s:.6g}",
                "muon_momentum": f"{muon_momentum:.4f}",
                "grad_norm": f"{gnorm:.4f}" if gnorm >= 0 else "",
                "step_time_ms": f"{dt*1000:.1f}",
                "tokens_per_sec": f"{tps:.0f}",
                "train_tokens": train_tokens,
                "val_tokens": val_tokens_count,
                "total_tokens": train_tokens + val_tokens_count,
                "train_time_s": f"{training_time_s:.1f}",
                "wall_time_s": f"{time.time() - wall_t0:.1f}",
                "peak_mem_mib": get_peak_mem_mib(),
            })

            if wandb_run is not None and ((step + 1) % args.wandb_log_every == 0):
                wandb.log({
                    "global_step": step + 1,
                    "loss/train": train_loss,
                    "train/loss": train_loss,
                    "train/grad_norm": gnorm if gnorm >= 0 else None,
                    "lr/embed": lr_e,
                    "lr/head": lr_h,
                    "lr/muon": lr_m,
                    "lr/scalar": lr_s,
                    "train/window_tokens": window_tokens,
                    "train/window_blocks": sw_blocks,
                    "perf/step_time_ms": dt * 1000.0,
                    "perf/tokens_per_sec": tps,
                    "mem/peak_mib": get_peak_mem_mib(),
                })

        # Periodic checkpoint
        if master_process and checkpoint_every > 0 and (step == 0 or (step + 1) % checkpoint_every == 0):
            p = save_training_checkpoint(step + 1 if step > 0 else 0, raw_model, optimizers, args.output_dir, val_loss)
            log_print(f"💾 Checkpoint: {p}")

        step += 1

    # Done
    wall = time.time() - wall_t0
    avg_tps = (train_tokens / training_time_s) if training_time_s > 0 else 0.0
    log_print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")
    log_print(f"steps {step:,} | train_tokens {train_tokens:,} | train_time {training_time_s:.1f}s | wall {wall:.1f}s")
    log_print(f"avg throughput {avg_tps:,.0f} tok/s | final val_loss {val_loss:.6f}")
    if device_type == "cuda":
        log_print(f"peak mem {get_peak_mem_mib()} MiB")

    destroy_distributed(dist_env)

    if master_process and wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
