# ============================================================
# CELL 1 — Imports + small utilities
# ============================================================
import os, math, glob, time, json, csv
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from typing import Optional, Dict, List, Tuple
from scale_arch_utils import (
    build_sparse_endpoint_pattern,
    default_batch_sizes_for_profile,
    default_skip_attn_layer,
    default_vte_endpoint_k,
    resolve_model_dims,
    validate_dims,
)

import numpy as np
import torch
import sys

HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

# Optional wandb (disabled by default in config)
try:
    import wandb
except Exception:
    wandb = None

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, "| VRAM(GB):", round(props.total_memory / 1024**3, 1))


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_peak_mem_mib() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() // 1024 // 1024)
    return 0

def get_free_total_mib() -> Tuple[int, int]:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return int(free // 1024 // 1024), int(total // 1024 // 1024)
    return 0, 0

def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def make_run_dir(base_out: str, tag: str) -> str:
    os.makedirs(base_out, exist_ok=True)
    # sanitize a bit
    safe = tag.replace("/", "_").replace(" ", "")
    return os.path.join(base_out, safe)

class TeeLogger:
    """Context manager to tee print statements to both console and a log file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = None
        self.original_stdout = None
        
    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else ".", exist_ok=True)
        self.log_file = open(self.log_path, 'w', buffering=1)  # Line buffered
        self.original_stdout = __builtins__.print
        
        # Override print to write to both
        def tee_print(*args, **kwargs):
            # Print to console
            self.original_stdout(*args, **kwargs)
            # Print to file
            kwargs_copy = kwargs.copy()
            kwargs_copy['file'] = self.log_file
            kwargs_copy['flush'] = True
            self.original_stdout(*args, **kwargs_copy)
        
        __builtins__.print = tee_print
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        __builtins__.print = self.original_stdout
        if self.log_file:
            self.log_file.close()
        return False


def configure_attention_backend(use_cudnn_attn: bool) -> None:
    if not torch.cuda.is_available():
        return
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        )
        if use_cudnn_attn:
            enable_cudnn_sdp(True)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            print("[INFO] Enabled CUDNN attention.")
        else:
            # choose defaults; you can switch these if you want Flash
            enable_cudnn_sdp(False)
            enable_flash_sdp(True)
            enable_mem_efficient_sdp(True)
            enable_math_sdp(True)
            print("[INFO] Enabled Flash/MemEff attention backends.")
    except Exception as e:
        print(f"[WARN] Could not configure SDP backends: {e}")


# ============================================================
# CELL 2 — Configs
# ============================================================
@dataclass
class TrainConfig:
    # I/O
    train_pattern: str
    val_pattern: str
    output_dir: str

    # Model
    vocab_size: int = 50304
    model_profile: str = "d12"
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    skip_attn_layer: int = -1
    vte_endpoint_k: int = 0

    # Batching (IMPORTANT: device_batch_size must stay 1 due to flex-attn assert)
    batch_size: int = 8              # global batch in sequences
    device_batch_size: int = 1
    sequence_length: int = 65536
    block_size: int = 128

    # Optimizer LRs (4-way)
    embed_lr: float = 0.6
    head_lr: float = 0.008
    muon_lr: float = 0.04
    scalar_lr: float = 0.04

    # Muon momentum schedule
    muon_momentum_init: float = 0.85
    muon_momentum_final: float = 0.95
    muon_momentum_warmup_steps: int = 300

    # LR schedule (step-based; ok because we token-cap pilots)
    warmup_frac: float = 0.0
    warmdown_frac: float = 0.1

    # Training loop control
    num_iterations: int = 20000
    grad_clip: float = 1.0
    seed: int = 42
    loss_threshold: float = 3.3
    stop_mode: str = "const_loss"  # "const_loss" or "epoch"
    stop_epoch_frac: Optional[float] = None

    # Window warmup
    window_min: int = 64
    window_max: int = 1792
    window_warmup_steps: int = 4000

    # Eval + ckpt (val_tokens still used to define eval cost)
    val_loss_every: int = 100
    val_tokens: int = 10_485_760
    checkpoint_every: int = 400

    # Token-based eval cadence (batch-invariant):
    # Baseline (bs=8, T=65536, eval every 100 steps) => 52,428,800 tokens.
    eval_interval_tokens: int = 52_428_800

    # Numerics
    dtype: str = "bfloat16"          # float32|bfloat16|float16
    compile: bool = True
    tensorcores: bool = True
    use_cudnn_attn: bool = True

    # WandB
    wandb_project: str = "gpt2-dynamics"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "14_truncated_rope_sweep"
    wandb_mode: str = "disabled"     # online|offline|disabled
    wandb_log_every: int = 1


@dataclass
class ASHASweepConfig:
    """ASHA (Successive Halving) sweep config with power-law extrapolation."""
    batch_sizes: List[int] = None
    local_multipliers: Tuple[float, ...] = (0.5, 0.70710678, 1.0, 1.41421356, 2.0)
    
    # ASHA rung system: progressively larger token budgets
    # Each rung runs to the specified token count, then top 1/eta are promoted
    rungs: Tuple[int, ...] = (50_000_000, 150_000_000, 500_000_000)  # 50M → 150M → 500M
    eta: int = 3  # Halving factor: keep top 1/eta at each rung
    
    # Eval interval during sweeps (for building val_history)
    eval_interval_tokens: int = 20_000_000  # 20M tokens between evals
    
    # Power-law extrapolation target: predict loss at this token count
    extrapolation_target: int = 1_000_000_000  # 1B tokens
    
    # Minimum eval points needed for power-law fit
    min_evals_for_fit: int = 3
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [4, 8, 16]


def fit_power_law(tokens: List[float], losses: List[float], target_tokens: float) -> Tuple[float, dict]:
    """
    Fit loss(t) = a * t^(-b) + c (power law with offset) and predict at target_tokens.
    
    Based on Domhan et al. 2015 (IJCAI): "Speeding Up Automatic Hyperparameter 
    Optimization of Deep Neural Networks by Extrapolation of Learning Curves"
    
    Returns:
        (predicted_loss, fit_info) where fit_info contains a, b, c, r2, or error info
    """
    from scipy.optimize import curve_fit
    
    # Filter out data points with t <= 0 (power-law is undefined at t=0)
    filtered = [(t, l) for t, l in zip(tokens, losses) if t > 0]
    if len(filtered) < 3:
        # Not enough valid points for fitting
        return losses[-1], {"error": "insufficient_positive_t_points", "fallback": True}
    
    tokens_arr = np.array([t for t, l in filtered], dtype=np.float64)
    losses_arr = np.array([l for t, l in filtered], dtype=np.float64)
    
    def power_law(t, a, b, c):
        return a * np.power(t, -b) + c
    
    # Initial parameter guesses
    c0 = min(losses_arr) * 0.9  # Asymptote slightly below observed minimum
    a0 = (losses_arr[0] - c0) * (tokens_arr[0] ** 0.5)  # Scale estimate
    b0 = 0.5  # Typical decay exponent
    
    try:
        # Fit with bounds to ensure physical constraints
        # a > 0, 0 < b < 2 (reasonable decay range), c > 0
        popt, pcov = curve_fit(
            power_law, tokens_arr, losses_arr,
            p0=[a0, b0, c0],
            bounds=([1e-10, 0.01, 0.0], [np.inf, 2.0, min(losses_arr) + 1.0]),
            maxfev=5000
        )
        a, b, c = popt
        
        # Predict at target
        predicted = power_law(target_tokens, a, b, c)
        
        # Compute R² goodness of fit
        ss_res = np.sum((losses_arr - power_law(tokens_arr, *popt)) ** 2)
        ss_tot = np.sum((losses_arr - np.mean(losses_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return predicted, {"a": float(a), "b": float(b), "c": float(c), "r2": float(r2)}
    
    except Exception as e:
        # Fallback: return last observed loss (no penalty, just assume constant)
        return losses[-1], {"error": str(e), "fallback": True}


def tokens_per_step(cfg: TrainConfig) -> int:
    # 1-GPU, device_batch_size=1 => tokens/step = batch_size * sequence_length
    return int(cfg.batch_size * cfg.sequence_length)

def scale_steps_by_tokens(base_steps: int, base_cfg: TrainConfig, new_cfg: TrainConfig) -> int:
    base_tps = tokens_per_step(base_cfg)
    new_tps  = tokens_per_step(new_cfg)
    tok_budget = base_steps * base_tps
    return max(1, int(round(tok_budget / new_tps)))

def make_cfg_for_batch(base_cfg: TrainConfig, new_batch_size: int) -> TrainConfig:
    c = TrainConfig(**asdict(base_cfg))
    c.batch_size = int(new_batch_size)

    # token-normalize warmups
    c.window_warmup_steps = scale_steps_by_tokens(base_cfg.window_warmup_steps, base_cfg, c)
    c.muon_momentum_warmup_steps = scale_steps_by_tokens(base_cfg.muon_momentum_warmup_steps, base_cfg, c)

    # token-based eval interval should stay constant across batch sizes (already is)
    # but keep it explicit:
    c.eval_interval_tokens = base_cfg.eval_interval_tokens
    return c


# ============================================================
# CELL 3 — Muon optimizer
# ============================================================
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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


# ============================================================
# CELL 4 — Model (FlexAttention + VTE)
# ============================================================
try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    flex_attention = torch.compile(flex_attention, dynamic=False)
except Exception as e:
    if not HELP_REQUESTED:
        raise RuntimeError("This script requires torch.nn.attention.flex_attention (flex_attention + BlockMask).") from e
    flex_attention = None
    BlockMask = None

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

class Rotary(nn.Module):
    """
    Truncated/"weird" RoPE as in the reference script:
      inv_freq = (1/1024) ** linspace(0,1, steps=dim//4), then padded with zeros.

    This module APPLY-rotates x and returns the rotated tensor (not cos/sin).
    Expected x shape: [B, T, nH, Hd], where Hd is even.
    """
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        assert head_dim % 4 == 0, f"reference truncated RoPE expects head_dim divisible by 4, got {head_dim}"

        inv_freq = (1.0 / 1024.0) ** torch.linspace(
            0.0, 1.0, steps=head_dim // 4, dtype=torch.float32
        )
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(head_dim // 4)])  # -> head_dim//2

        t = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
        theta = torch.einsum("i, j -> ij", t, inv_freq)      # [max_seq_len, head_dim//2]

        # Buffers; they will move with .to(device) on the module
        self.cos = nn.Buffer(theta.cos(), persistent=False)  # [max_seq_len, head_dim//2]
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, nH, Hd]
        T = x.size(1)
        cos = self.cos[None, :T, None, :]  # [1, T, 1, Hd/2]
        sin = self.sin[None, :T, None, :]

        # Do rotation in fp32 for numeric stability, then cast back
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)  # each [B,T,nH,Hd/2]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    """
    FlexAttention requires batch size 1.
    This matches the reference script behavior:
      - Q/K RMSNorm
      - RoPE on Q/K
      - V is mixed with VE via learnable lambdas/mix
      - output projection is zero-init
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)

        self.mix = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))  # [base_v, ve_v]

        self.rotary = Rotary(self.head_dim, max_seq_len=max_seq_len)

        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: Tensor, ve: Optional[Tensor], block_mask: BlockMask) -> Tensor:
        # x: [1, T, dim]
        B, T, C = x.shape
        assert B == 1, f"FlexAttention path assumes batch size 1, got B={B}"
        assert C == self.dim

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)

        if ve is None:
            v = self.mix[0] * v
        else:
            # accept ve as [T, C] or [1, T, C]
            if ve.ndim == 2:
                ve = ve[None, :, :]
            assert ve.shape == (B, T, self.dim), f"Expected ve {(B,T,self.dim)}, got {tuple(ve.shape)}"
            ve = ve.view(B, T, self.num_heads, self.head_dim).to(dtype=v.dtype)
            v = self.mix[0] * v + self.mix[1] * ve

        # QK norm then rotary
        q = apply_norm(q)
        k = apply_norm(k)
        q = self.rotary(q)
        k = self.rotary(k)

        # FlexAttention expects [B, nH, T, Hd]
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.dim)
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
    def __init__(self, dim: int, n_head: int, layer_idx: int, max_seq_len: int, skip_attn_layer: int = 7):
        super().__init__()
        self.layer_idx = layer_idx
        self.skip_attn_layer = skip_attn_layer

        self.attn = None if layer_idx == skip_attn_layer else CausalSelfAttention(dim, n_head, max_seq_len=max_seq_len)
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))  # mix current x and x0

    def forward(self, x: Tensor, ve: Optional[Tensor], x0: Tensor, block_mask: BlockMask) -> Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(apply_norm(x), ve, block_mask)
        x = x + self.mlp(apply_norm(x))
        return x


class ValueTokenEmbedding(nn.Module):
    """
    Sparse VTE: 3 embedding tables, with a fixed per-layer pattern.
    Returns: list length n_layer of [T, C] or None.
    """
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, endpoint_k: int = 0):
        super().__init__()
        self.n_layer = n_layer
        self.endpoint_k = endpoint_k
        self.emb = nn.ModuleList([nn.Embedding(vocab_size, n_embd) for _ in range(3)])

    def forward(self, tokens_1d: Tensor) -> List[Optional[Tensor]]:
        ve0 = self.emb[0](tokens_1d)  # [T, C]
        ve1 = self.emb[1](tokens_1d)
        ve2 = self.emb[2](tokens_1d)

        pattern = build_sparse_endpoint_pattern(ve0, ve1, ve2, self.n_layer, self.endpoint_k)
        assert len(pattern) == self.n_layer, f"VE pattern len {len(pattern)} != n_layer {self.n_layer}"
        return pattern


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    seq_len: int = 65536      # IMPORTANT: drive RoPE + mask sizes; set from args.sequence_length
    block_size: int = 128     # FlexAttention KV block size (fixed at 128 in the reference)
    skip_attn_layer: int = -1
    vte_endpoint_k: int = 0


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers, dtype=torch.float32))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([
                Block(config.n_embd, config.n_head, i, max_seq_len=config.seq_len, skip_attn_layer=config.skip_attn_layer)
                for i in range(config.n_layer)
            ]),
        ))
        self.value_embeds = ValueTokenEmbedding(
            config.vocab_size,
            config.n_embd,
            config.n_layer,
            endpoint_k=config.vte_endpoint_k,
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def _make_block_mask(self, tokens_1d: Tensor, sliding_window_num_blocks: Tensor) -> BlockMask:
        BLOCK = self.config.block_size
        T = tokens_1d.numel()
        assert T % BLOCK == 0, f"T={T} must be divisible by BLOCK_SIZE={BLOCK}"
        n_blocks = T // BLOCK

        # doc ids per token
        docs = (tokens_1d == 50256).cumsum(0)

        docs_low  = docs.view(n_blocks, BLOCK)[:, 0].contiguous()
        docs_high = docs.view(n_blocks, BLOCK)[:, -1].contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

        sw = torch.as_tensor(sliding_window_num_blocks, dtype=torch.int32, device=tokens_1d.device).clamp_min(1)

        block_idx = torch.arange(n_blocks, dtype=torch.int32, device=tokens_1d.device)
        q_idx = block_idx[:, None]
        kv_idx = block_idx[None, :]

        causal_bm = q_idx >= kv_idx
        causal_full_bm = q_idx > kv_idx
        window_bm = (q_idx - kv_idx) < sw
        window_full_bm = window_bm

        # overlap in doc ranges (block-level)
        document_bm = (docs_low[:, None] <= docs_high[None, :]) & (docs_low[None, :] <= docs_high[:, None])
        document_full_bm = (docs_low[:, None] == docs_high[None, :]) & (docs_low[None, :] == docs_high[:, None])

        nonzero_bm = causal_bm & window_bm & document_bm
        full_bm = causal_full_bm & window_full_bm & document_full_bm

        def dense_to_ordered(dense_mask: Tensor):
            nb = dense_mask.sum(dim=-1, dtype=torch.int32)
            idx = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return nb[None, None].contiguous(), idx[None, None].contiguous()

        kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)

        return BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices,
            full_kv_num_blocks, full_kv_indices,
            BLOCK_SIZE=BLOCK,
            mask_mod=mask_mod,
        )

    def forward(self, inputs: Tensor, targets: Tensor, sliding_window_num_blocks: Tensor) -> Tensor:
        # inputs/targets: [B,T] or [T], B must be 1
        if inputs.ndim == 2:
            assert inputs.size(0) == 1
            inputs = inputs[0]
        if targets.ndim == 2:
            assert targets.size(0) == 1
            targets = targets[0]
        assert inputs.ndim == 1 and targets.ndim == 1
        assert inputs.numel() == targets.numel()

        block_mask = self._make_block_mask(inputs, sliding_window_num_blocks)

        x = self.transformer.wte(inputs[None])  # [1,T,C]
        x = apply_norm(x)
        x0 = x

        ve_all = self.value_embeds(inputs)  # list len n_layer

        skips = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_all[i], x0, block_mask)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            j = self.num_encoder_layers + i
            x = self.transformer.h[j](x, ve_all[j], x0, block_mask)

        x = apply_norm(x)
        logits = self.lm_head(x)
        
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ============================================================
# CELL 5 — Data loader (FineWeb / raw / npy)
# ============================================================
FW_MAGIC = 20240520
HEADER_BYTES = 256 * 4
_RAW_DTYPE_CACHE: Dict[str, np.dtype] = {}

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
    if size % 2 == 0: cand.append(np.uint16)
    if size % 4 == 0: cand.append(np.uint32)
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
    fmt, _ = _detect_format(path)
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
    fmt, _ = _detect_format(path)
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
    def __init__(self, filename_pattern, B, T, process_rank=0, num_processes=1):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        self.ntok_total = sum(_peek_data_shard(f) for f in self.files)
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


# ============================================================
# CELL 6 — Training: one run + val slope per token
# ============================================================
def train_one_run(
    cfg: TrainConfig,
    run_dir: str,
    max_train_tokens: Optional[int],
    token_based_eval: bool = True,
    max_wall_s: Optional[float] = None,
    slope_K: int = 5,
    eval_interval_tokens: Optional[int] = None,  # Override eval interval for pilots
) -> dict:
    assert cfg.device_batch_size == 1, "FlexAttention model asserts B==1 in forward."
    assert cfg.batch_size % cfg.device_batch_size == 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    
    # Use override eval interval if provided, otherwise use cfg default
    effective_eval_interval = eval_interval_tokens if eval_interval_tokens is not None else cfg.eval_interval_tokens

    # Check for existing run completion
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_csv) and max_train_tokens is not None:
        try:
            rows = []
            with open(metrics_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            
            if rows:
                last_row = rows[-1]
                done_tokens = int(float(last_row["train_tokens"]))
                if done_tokens >= max_train_tokens:
                    print(f"[INFO] Found existing run in {run_dir} with {done_tokens} tokens. Skipping training.")
                    
                    val_history = []
                    seen_val_tokens = -1
                    for r in rows:
                        vt = int(float(r["val_tokens_total"]))
                        if vt > seen_val_tokens:
                             val_loss = float(r["last_val"])
                             if not math.isnan(val_loss):
                                 t = int(float(r["train_tokens"]))
                                 val_history.append((t, val_loss))
                             seen_val_tokens = vt
                    
                    return {
                        "last_val": float(last_row["last_val"]),
                        "train_loss": float(last_row["train_loss"]),
                        "train_tokens": done_tokens,
                        "val_history": val_history,
                    }
        except Exception as e:
            print(f"[WARN] Failed to read existing metrics from {metrics_csv}: {e}. Restarting run.")

    set_seed(cfg.seed)
    os.makedirs(run_dir, exist_ok=True)
    save_json(os.path.join(run_dir, "cfg.json"), asdict(cfg))

    # precision & backend
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    if cfg.tensorcores and device_type == "cuda":
        torch.set_float32_matmul_precision("high")
    configure_attention_backend(cfg.use_cudnn_attn)

    # model
    raw_model = GPT(GPTConfig(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        seq_len=cfg.sequence_length,
        block_size=cfg.block_size,
        skip_attn_layer=cfg.skip_attn_layer,
        vte_endpoint_k=cfg.vte_endpoint_k,
    )).to(device)
    raw_model.train()

    if cfg.compile and device_type == "cuda":
        model = torch.compile(raw_model)
    else:
        model = raw_model

    # data
    train_loader = DistributedDataLoader(cfg.train_pattern, cfg.device_batch_size, cfg.sequence_length)
    val_loader = DistributedDataLoader(cfg.val_pattern, cfg.device_batch_size, cfg.sequence_length) if cfg.val_pattern else None

    grad_accum_steps = cfg.batch_size // cfg.device_batch_size
    step_tokens = cfg.batch_size * cfg.sequence_length
    print(f"[INFO] step_tokens = {step_tokens:,}")

    # optimizers (use raw_model, not compiled model)
    # ve_params = list(raw_model.value_embeds.parameters())
    # block_params = list(raw_model.transformer.h.parameters())
    
    ve_params = list(raw_model.value_embeds.parameters())
    block_params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in block_params if p.ndim == 2]
    scalar_params = [p for p in block_params if p.ndim < 2]
    scalar_params.append(raw_model.skip_weights)

    opt_embed = torch.optim.Adam(
        ve_params + [raw_model.transformer.wte.weight],
        lr=cfg.embed_lr, betas=(0.8, 0.95), fused=(device_type=='cuda')
    )
    opt_head  = torch.optim.Adam(
        [raw_model.lm_head.weight],
        lr=cfg.head_lr, betas=(0.9,0.95), fused=(device_type=='cuda')
    )
    opt_muon  = Muon(matrix_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum_init)
    opt_scalar= torch.optim.Adam(
        scalar_params,
        lr=cfg.scalar_lr, betas=(0.8, 0.95), fused=(device_type=='cuda')
    )
    optimizers = [opt_embed, opt_head, opt_muon, opt_scalar]

    # LR schedulers (step-based)
    total_steps = cfg.num_iterations
    warmup_steps = int(round(cfg.warmup_frac * total_steps))
    warmdown_steps = int(round(cfg.warmdown_frac * total_steps))
    plateau_steps = max(0, total_steps - warmup_steps - warmdown_steps)

    def lr_mult(it):
        it = min(it, total_steps)
        if warmup_steps > 0 and it < warmup_steps:
            return (it+1)/warmup_steps
        decay_start = warmup_steps + plateau_steps
        if warmdown_steps <= 0 or it < decay_start:
            return 1.0
        decay_ratio = (total_steps - it) / max(1, warmdown_steps)
        return max(0.0, decay_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_mult) for opt in optimizers]

    # validation setup
    val_steps_per_eval = None
    if val_loader is not None:
        assert cfg.val_tokens % (cfg.device_batch_size * cfg.sequence_length) == 0
        val_steps_per_eval = cfg.val_tokens // (cfg.device_batch_size * cfg.sequence_length)

    # logging
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    val_history: List[Tuple[int,float]] = []

    with open(metrics_csv, "w", newline="") as f_csv:
        fieldnames = [
            "timestamp","step","train_loss","best_val","last_val",
            "train_tokens","val_tokens_total",
            "lr_embed","lr_head","lr_muon","lr_scalar","muon_momentum","attn_blocksize",
            "step_time_ms","tok_per_s","peak_mem_mib"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        train_tokens = 0
        val_tokens_total = 0
        step = 0
        best_val_loss = float("inf")
        last_val_loss = float("nan")

        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)

        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        wall_t0 = time.time()
        skip_warmup_steps = 10
        train_time_s = 0.0

        # token-based eval schedule: use effective_eval_interval (may be overridden for pilots)
        next_eval_at = 0 if not token_based_eval else effective_eval_interval

        while True:
            if max_wall_s is not None and (time.time()-wall_t0) > max_wall_s:
                break
            if max_train_tokens is not None and train_tokens >= max_train_tokens:
                break
            if step >= cfg.num_iterations:
                break

            # window warmup (step-based, but cfg.window_warmup_steps is token-scaled already)
            wfrac = min(step / max(1,cfg.window_warmup_steps), 1.0)
            window_size = cfg.window_min + wfrac*(cfg.window_max-cfg.window_min)
            
            # Convert to blocks
            sw_blocks = int(window_size // 128)
            if sw_blocks < 1: sw_blocks = 1
            sliding_window_num_blocks = torch.tensor(sw_blocks, dtype=torch.int32, device=device)
            # Log tokens for consistency with other logs
            attn_blocksize = int(sw_blocks * 128)

            # muon momentum warmup (step-based, but cfg.muon_momentum_warmup_steps is token-scaled already)
            mfrac = min(step / max(1,cfg.muon_momentum_warmup_steps), 1.0)
            muon_mom = (1-mfrac)*cfg.muon_momentum_init + mfrac*cfg.muon_momentum_final
            opt_muon.param_groups[0]["momentum"] = float(muon_mom)

            # evaluation decision
            do_eval = False
            if val_loader is not None:
                if token_based_eval:
                    if train_tokens >= next_eval_at:
                        do_eval = True
                else:
                    if (cfg.val_loss_every > 0) and (step % cfg.val_loss_every == 0):
                        do_eval = True

            if do_eval and val_loader is not None:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    sum_val = torch.tensor(0.0, device=device)
                    for _ in range(val_steps_per_eval):
                        xv,yv = val_loader.next_batch()
                        xv,yv = xv.to(device), yv.to(device)
                        with ctx:
                            sum_val += model(xv,yv,sliding_window_num_blocks)
                    last_val_loss = float((sum_val/ max(1,val_steps_per_eval)).item())

                val_history.append((train_tokens, last_val_loss))
                val_tokens_total += cfg.val_tokens

                if token_based_eval:
                    next_eval_at += effective_eval_interval

                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss

                # Stopping criteria - mutually exclusive modes
                if cfg.stop_mode == "const_loss":
                    # Mode 1: Stop by constant loss threshold
                    if last_val_loss < cfg.loss_threshold and step > 0:
                        model.train()
                        break
                elif cfg.stop_mode == "epoch":
                    # Mode 2: Stop by epoch fraction
                    if cfg.stop_epoch_frac is not None:
                        # Calculate epoch fraction
                        tokens_per_epoch = train_loader.ntok_total
                        epoch_frac = train_tokens / tokens_per_epoch if tokens_per_epoch > 0 else 0.0
                        if epoch_frac >= cfg.stop_epoch_frac:
                            model.train()
                            break

                model.train()

            # training step
            model.train()
            t0 = time.perf_counter()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            step_losses=[]
            for _ in range(grad_accum_steps):
                with ctx:
                    l = model(x,y,sliding_window_num_blocks)
                    (l/grad_accum_steps).backward()
                    step_losses.append(l.detach())
                x,y = train_loader.next_batch()
                x,y = x.to(device), y.to(device)

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)

            for opt,sch in zip(optimizers,schedulers):
                opt.step()
                sch.step()

            dt = time.perf_counter()-t0
            if step>=skip_warmup_steps:
                train_time_s += dt
            train_tokens += step_tokens

            train_loss = float(torch.stack(step_losses).mean().item())
            tok_per_s = step_tokens / max(dt,1e-9)

            writer.writerow({
                "timestamp": now_ts(),
                "step": step,
                "train_loss": train_loss,
                "best_val": best_val_loss,
                "last_val": last_val_loss,
                "train_tokens": train_tokens,
                "val_tokens_total": val_tokens_total,
                "lr_embed": opt_embed.param_groups[0]["lr"],
                "lr_head":  opt_head.param_groups[0]["lr"],
                "lr_muon":  opt_muon.param_groups[0]["lr"],
                "lr_scalar":opt_scalar.param_groups[0]["lr"],
                "muon_momentum": muon_mom,
                "attn_blocksize": attn_blocksize,
                "step_time_ms": dt*1000.0,
                "tok_per_s": tok_per_s,
                "peak_mem_mib": get_peak_mem_mib(),
            })
            f_csv.flush()
            step += 1

        # Force a final validation at the end of the rung (if we have a val_loader)
        if val_loader is not None:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                sum_val = torch.tensor(0.0, device=device)
                for _ in range(val_steps_per_eval):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device), yv.to(device)
                    with ctx:
                        sum_val += model(xv, yv, sliding_window_num_blocks)
                final_val_loss = float((sum_val / max(1, val_steps_per_eval)).item())
            
            if len(val_history) == 0 or val_history[-1][0] != train_tokens:
                val_history.append((train_tokens, final_val_loss))
                val_tokens_total += cfg.val_tokens
            
            last_val_loss = final_val_loss
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
            
            model.train()

        # slope estimate on last K evals
        if len(val_history) >= 2:
            use = val_history[-min(slope_K, len(val_history)):]
            slopes = []
            for i in range(1, len(use)):
                dtok = use[i][0] - use[i-1][0]
                dloss = use[i][1] - use[i-1][1]
                if dtok > 0:
                    slopes.append(dloss / dtok)
            avg_slope = float(sum(slopes)/len(slopes)) if len(slopes)>0 else float("nan")
        else:
            avg_slope = float("nan")

        summary = dict(
            run_dir=run_dir,
            steps=step,
            train_tokens=train_tokens,
            val_tokens_total=val_tokens_total,
            best_val=best_val_loss,
            last_val=last_val_loss,
            val_slope_per_token=avg_slope,
            eval_count=len(val_history),
            val_history=val_history,  # For power-law extrapolation
            wall_s=(time.time()-wall_t0),
            avg_tok_s=(train_tokens/(train_time_s if train_time_s>0 else 1.0)),
        )
        save_json(os.path.join(run_dir,"summary.json"), summary)
        return summary


# ============================================================
# CELL 7 — LR candidate generation + sweep driver
# ============================================================
def theory_lr_scales(new_bs: int, base_bs: int) -> Dict[str, float]:
    ratio = new_bs / base_bs
    return {"sqrt": math.sqrt(ratio), "linear": ratio}

def build_lr_candidates(new_bs: int, base_bs: int, local_mults: Tuple[float, ...]) -> List[Tuple[str, float]]:
    scales = theory_lr_scales(new_bs, base_bs)
    cands = []
    for name, center in scales.items():
        for m in local_mults:
            val = center * m
            tag = f"{name}*{m:.3g}"
            cands.append((tag, val))
    # dedup near-equal
    uniq = {}
    for tag, v in cands:
        key = round(v, 6)
        if key not in uniq:
            uniq[key] = (tag, v)
    out = list(uniq.values())
    out.sort(key=lambda x: x[1])
    return out

def apply_lr_mult(cfg: TrainConfig, lr_mult: float) -> TrainConfig:
    c = TrainConfig(**asdict(cfg))
    c.embed_lr  = cfg.embed_lr  * lr_mult
    c.head_lr   = cfg.head_lr   * lr_mult
    c.muon_lr   = cfg.muon_lr   * lr_mult
    c.scalar_lr = cfg.scalar_lr * lr_mult
    return c

def run_asha_sweep(base_cfg: TrainConfig, sweep: ASHASweepConfig) -> List[dict]:
    """
    ASHA (Asynchronous Successive Halving) sweep with power-law learning curve extrapolation.
    
    Algorithm:
    1. For each batch_size, generate LR candidates
    2. Run all candidates through successive rungs (e.g., 50M → 150M → 500M tokens)
    3. At each rung, fit power-law to val_history and predict loss at extrapolation_target
    4. Keep top 1/eta candidates based on predicted loss
    5. Continue survivors to the next rung
    6. Return the final survivors as best configs
    
    Reference: Li et al. 2018 "A System for Massively Parallel Hyperparameter Tuning"
    """
    all_results: List[dict] = []
    base_bs = base_cfg.batch_size
    
    for bs in sweep.batch_sizes:
        bs_cfg = make_cfg_for_batch(base_cfg, bs)
        candidates = build_lr_candidates(bs, base_bs, sweep.local_multipliers)
        
        print("\n" + "="*90)
        print(f"[ASHA] batch_size={bs} | tokens/step={tokens_per_step(bs_cfg):,}")
        print(f"       Rungs: {[f'{r/1e6:.0f}M' for r in sweep.rungs]}")
        print(f"       Halving factor eta={sweep.eta}")
        print(f"       Extrapolation target: {sweep.extrapolation_target/1e9:.1f}B tokens")
        print(f"       Candidates: {len(candidates)}")
        print("="*90)
        
        # Active candidates: list of (tag, lr_mult, cfg, run_dir, summary)
        active = []
        for cand_tag, lr_mult in candidates:
            trial_cfg = apply_lr_mult(bs_cfg, lr_mult)
            trial_cfg.seed = base_cfg.seed
            tag = f"bs{bs}_lr{lr_mult:.6g}_{cand_tag.replace('*','x')}"
            run_dir = make_run_dir(base_cfg.output_dir, tag)
            active.append({
                "tag": tag,
                "cand_tag": cand_tag,
                "lr_mult": lr_mult,
                "cfg": trial_cfg,
                "run_dir": run_dir,
                "summary": None,
                "predicted_loss": float("inf"),
                "fit_info": None,
            })
        
        # Process each rung
        for rung_idx, rung_tokens in enumerate(sweep.rungs):
            n_active = len(active)
            if n_active == 0:
                print(f"[WARN] No active candidates remaining at rung {rung_idx+1}")
                break
                
            print(f"\n{'─'*80}")
            print(f"[RUNG {rung_idx+1}/{len(sweep.rungs)}] {rung_tokens/1e6:.0f}M tokens | {n_active} candidates")
            print(f"{'─'*80}")
            
            # Run each active candidate to this rung's token budget
            for i, cand in enumerate(active):
                print(f"\n[{i+1}/{n_active}] {cand['tag']}")
                
                summary = train_one_run(
                    cfg=cand["cfg"],
                    run_dir=cand["run_dir"],
                    max_train_tokens=rung_tokens,
                    token_based_eval=(cand["cfg"].stop_mode == "epoch"),
                    max_wall_s=None,
                    slope_K=5,
                    eval_interval_tokens=sweep.eval_interval_tokens,
                )
                cand["summary"] = summary
                
                # Power-law extrapolation
                val_history = summary.get("val_history", [])
                if len(val_history) >= sweep.min_evals_for_fit:
                    tokens_list = [float(t) for t, _ in val_history]
                    losses_list = [float(l) for _, l in val_history]
                    predicted, fit_info = fit_power_law(
                        tokens_list, losses_list, 
                        target_tokens=float(sweep.extrapolation_target)
                    )
                    cand["predicted_loss"] = predicted
                    cand["fit_info"] = fit_info
                    print(f"    Val loss: {summary['last_val']:.4f} | "
                          f"Predicted @{sweep.extrapolation_target/1e9:.0f}B: {predicted:.4f} | "
                          f"R²={fit_info.get('r2', 'N/A'):.3f}" if isinstance(fit_info.get('r2'), (int, float)) else
                          f"    Val loss: {summary['last_val']:.4f} | Fit failed, using fallback")
                else:
                    # Not enough evals - penalize with high predicted loss
                    cand["predicted_loss"] = summary.get("last_val", float("inf")) * 1.2
                    cand["fit_info"] = {"error": "insufficient_evals", "eval_count": len(val_history)}
                    print(f"    Val loss: {summary['last_val']:.4f} | "
                          f"Insufficient evals ({len(val_history)} < {sweep.min_evals_for_fit})")
            
            # === SORTING OPTION 1: Sort by predicted loss (power-law extrapolation) ===
            # active.sort(key=lambda x: x["predicted_loss"])
            # print(f"\n[PROMOTION] Keeping top {n_promote}/{len(active)} by predicted loss @{sweep.extrapolation_target/1e9:.0f}B")
            # for j, cand in enumerate(active[:n_promote]):
            #     print(f"    {j+1}. {cand['tag']}: predicted={cand['predicted_loss']:.4f}")
            
            # === SORTING OPTION 2: Sort by current validation loss ===
            active.sort(key=lambda x: x["summary"]["last_val"])
            n_promote = max(1, len(active) // sweep.eta)
            print(f"\n[PROMOTION] Keeping top {n_promote}/{len(active)} by current val loss")
            for j, cand in enumerate(active[:n_promote]):
                print(f"    {j+1}. {cand['tag']}: val_loss={cand['summary']['last_val']:.4f}")
            
            # Add eliminated candidates to results
            for cand in active[n_promote:]:
                result = cand["summary"].copy() if cand["summary"] else {}
                result.update({
                    "batch_size": bs,
                    "lr_mult": cand["lr_mult"],
                    "cand_tag": cand["cand_tag"],
                    "embed_lr": cand["cfg"].embed_lr,
                    "head_lr": cand["cfg"].head_lr,
                    "muon_lr": cand["cfg"].muon_lr,
                    "scalar_lr": cand["cfg"].scalar_lr,
                    "predicted_loss": cand["predicted_loss"],
                    "fit_info": cand["fit_info"],
                    "final_rung": rung_idx + 1,
                    "promoted": False,
                })
                all_results.append(result)
            
            # Keep only promoted candidates
            active = active[:n_promote]
        
        # Final survivors
        print(f"\n{'='*80}")
        print(f"[FINAL] batch_size={bs} | Survivors after {len(sweep.rungs)} rungs:")
        print(f"{'='*80}")
        
        for cand in active:
            result = cand["summary"].copy() if cand["summary"] else {}
            result.update({
                "batch_size": bs,
                "lr_mult": cand["lr_mult"],
                "cand_tag": cand["cand_tag"],
                "embed_lr": cand["cfg"].embed_lr,
                "head_lr": cand["cfg"].head_lr,
                "muon_lr": cand["cfg"].muon_lr,
                "scalar_lr": cand["cfg"].scalar_lr,
                "predicted_loss": cand["predicted_loss"],
                "fit_info": cand["fit_info"],
                "final_rung": len(sweep.rungs),
                "promoted": True,
            })
            all_results.append(result)
            
            print(f"  ★ {cand['tag']}")
            print(f"    LR mult: {cand['lr_mult']:.4g}")
            print(f"    Final val loss: {cand['summary']['last_val']:.4f}")
            print(f"    Predicted @{sweep.extrapolation_target/1e9:.0f}B: {cand['predicted_loss']:.4f}")
            if cand["fit_info"] and "r2" in cand["fit_info"]:
                info = cand["fit_info"]
                print(f"    Power-law fit: a={info['a']:.4g}, b={info['b']:.4f}, c={info['c']:.4f}, R²={info['r2']:.4f}")
    
    # Save all results
    save_json(os.path.join(base_cfg.output_dir, "asha_sweep_results.json"), {"results": all_results})
    return all_results


# ============================================================
# CELL 8 — Main entry point
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ASHA LR Sweep for VTE GPT-2 with Power-Law Extrapolation")
    parser.add_argument("--train_pattern", type=str, 
                       default="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin",
                       help="Glob pattern for training data shards")
    parser.add_argument("--val_pattern", type=str, 
                       default="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin",
                       help="Glob pattern for validation data shards")
    parser.add_argument("--output_dir", type=str, default="lr_sweep/14_truncated_rope", help="Output directory")
    parser.add_argument("--sequence_length", type=int, default=65536, help="Tokens per sequence")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None, help="Batch sizes to sweep")
    parser.add_argument("--model_profile", type=str, default="d12", choices=["d12", "d24", "d36", "d48"])
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--skip_attn_layer", type=int, default=-1)
    parser.add_argument("--vte_endpoint_k", type=int, default=0)
    
    # ASHA rung settings
    parser.add_argument("--rungs", type=int, nargs="+", default=[50_000_000, 150_000_000, 500_000_000],
                       help="Token budgets for each ASHA rung (e.g., 50000000 150000000 500000000)")
    parser.add_argument("--eta", type=int, default=3, help="Halving factor (keep top 1/eta at each rung)")
    parser.add_argument("--eval_interval_tokens", type=int, default=20_000_000, 
                       help="Token interval between validation evaluations")
    
    # Extrapolation settings
    parser.add_argument("--extrapolation_target", type=int, default=1_000_000_000,
                       help="Token count at which to predict loss via power-law extrapolation")
    parser.add_argument("--min_evals_for_fit", type=int, default=3,
                       help="Minimum eval points needed for power-law fit")
    
    # Use parse_known_args() to ignore Jupyter's kernel arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown arguments (likely from Jupyter): {unknown}")

    n_layer, n_head, n_embd = resolve_model_dims(args, default=(12, 6, 768))
    validate_dims(n_layer, n_head, n_embd, require_even_layers=True)
    skip_attn_layer = args.skip_attn_layer if args.skip_attn_layer >= 0 else default_skip_attn_layer(n_layer)
    vte_endpoint_k = args.vte_endpoint_k if args.vte_endpoint_k > 0 else default_vte_endpoint_k(n_layer)
    if args.batch_sizes is None:
        args.batch_sizes = default_batch_sizes_for_profile(args.model_profile)
    
    BASE = TrainConfig(
        train_pattern=args.train_pattern,
        val_pattern=args.val_pattern,
        output_dir=args.output_dir,
        model_profile=args.model_profile,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        skip_attn_layer=skip_attn_layer,
        vte_endpoint_k=vte_endpoint_k,
        sequence_length=args.sequence_length,
        wandb_mode="disabled",
    )

    SWEEP = ASHASweepConfig(
        batch_sizes=args.batch_sizes,
        rungs=tuple(args.rungs),
        eta=args.eta,
        eval_interval_tokens=args.eval_interval_tokens,
        extrapolation_target=args.extrapolation_target,
        min_evals_for_fit=args.min_evals_for_fit,
    )

    # Create log file for the sweep
    log_path = os.path.join(args.output_dir, "asha_sweep_log.txt")
    
    # Wrap execution in TeeLogger to save all output
    with TeeLogger(log_path):
        print("\n" + "="*90)
        print("ASHA LR SWEEP CONFIGURATION")
        print("="*90)
        print(f"Batch sizes: {SWEEP.batch_sizes}")
        print(f"ASHA Rungs: {[f'{r/1e6:.0f}M' for r in SWEEP.rungs]}")
        print(f"Halving factor (eta): {SWEEP.eta} (keep top 1/{SWEEP.eta} at each rung)")
        print(f"Eval interval: {SWEEP.eval_interval_tokens:,} tokens")
        print(f"Extrapolation target: {SWEEP.extrapolation_target:,} tokens ({SWEEP.extrapolation_target/1e9:.1f}B)")
        print(f"Min evals for fit: {SWEEP.min_evals_for_fit}")
        print(f"Log file: {log_path}")
        print("="*90 + "\n")
        
        all_results = run_asha_sweep(BASE, SWEEP)

        # Summarize
        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            cols = [
                "batch_size", "lr_mult", "cand_tag", "best_val", "last_val", 
                "predicted_loss", "eval_count", "final_rung", "promoted",
                "train_tokens", "avg_tok_s",
                "embed_lr", "head_lr", "muon_lr", "scalar_lr", "run_dir"
            ]
            df = df[[c for c in cols if c in df.columns]].sort_values(
                ["batch_size", "promoted", "predicted_loss"], 
                ascending=[True, False, True],
                na_position="last"
            )
            print("\n" + "="*90)
            print("ASHA SWEEP RESULTS SUMMARY")
            print("="*90)
            print(df.to_string())
            
            # Save summary CSV
            summary_csv = os.path.join(args.output_dir, "asha_sweep_summary.csv")
            df.to_csv(summary_csv, index=False)
            print(f"\nSummary saved to: {summary_csv}")
            
            # Print best configs per batch size
            print("\n" + "="*90)
            print("BEST CONFIGS (Survivors)")
            print("="*90)
            survivors = df[df.get("promoted", False) == True] if "promoted" in df.columns else df
            for bs in sorted(df["batch_size"].unique()):
                bs_survivors = survivors[survivors["batch_size"] == bs]
                if len(bs_survivors) > 0:
                    best = bs_survivors.iloc[0]
                    print(f"\nbatch_size={bs}:")
                    print(f"  LR multiplier: {best['lr_mult']:.4g}")
                    print(f"  Actual LRs: embed={best['embed_lr']:.4g}, head={best['head_lr']:.4g}, "
                          f"muon={best['muon_lr']:.4g}, scalar={best['scalar_lr']:.4g}")
                    print(f"  Final val loss: {best['last_val']:.4f}")
                    print(f"  Predicted @{SWEEP.extrapolation_target/1e9:.0f}B: {best['predicted_loss']:.4f}")
                    
        except ImportError:
            print("[WARN] pandas not available, skipping DataFrame summary")
        except Exception as e:
            print(f"[WARN] Error in summary: {e}")
        
        print("\n" + "="*90)
        print("ASHA SWEEP COMPLETE")
        print("="*90)
    
    return all_results


if __name__ == "__main__":
    main()
