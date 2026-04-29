from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from .modarith_variants import (
        AttentionScaleTokenLM,
        BaselineTokenLM,
        FixedWindowTokenLM,
        RopeTokenLM,
        UntiedTokenLM,
        UnetTokenLM,
        ValueMixTokenLM,
    )
    from .variant_utils import parse_variant_tokens
except ImportError:
    from modarith_variants import (
        AttentionScaleTokenLM,
        BaselineTokenLM,
        FixedWindowTokenLM,
        RopeTokenLM,
        UntiedTokenLM,
        UnetTokenLM,
        ValueMixTokenLM,
    )
    from variant_utils import parse_variant_tokens


ARCHITECTURE_STAGE_ORDER = (
    "baseline",
    "rope",
    "untie_embed",
    "value_mix",
    "unet",
    "fixed_window",
    "attn_scale",
)


def resolve_modarith_stage(variant: str) -> str:
    tokens = parse_variant_tokens(variant)
    for stage in reversed(ARCHITECTURE_STAGE_ORDER):
        if stage in tokens:
            return stage
    return "baseline"


class LinearTokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_embed.weight

    def forward(self, x_tokens: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.tok_embed(x_tokens)
        logits = self.lm_head(h)
        if return_repr:
            return logits, h
        return logits, None


def build_modarith_model(
    track: str,
    vocab_size: int,
    seq_len: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_mult: int = 2,
    variant: str = "baseline",
    window_size: int = 0,
    attention_scale: Optional[float] = None,
    lm_head_softcap: float = 30.0,
) -> nn.Module:
    t = track.strip().lower()
    if t in {"b", "linear", "linear_lm"}:
        return LinearTokenLM(vocab_size=vocab_size, d_model=d_model)

    if t not in {"a", "transformer", "transformer_lm"}:
        raise ValueError(f"Unknown track: {track}")

    stage = resolve_modarith_stage(variant)
    if stage == "baseline":
        return BaselineTokenLM(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
        )
    if stage == "rope":
        return RopeTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
        )
    if stage == "untie_embed":
        return UntiedTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
        )
    if stage == "value_mix":
        return ValueMixTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            lm_head_softcap=lm_head_softcap,
        )
    if stage == "unet":
        return UnetTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            lm_head_softcap=lm_head_softcap,
        )
    if stage == "fixed_window":
        return FixedWindowTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            window_size=window_size,
            lm_head_softcap=lm_head_softcap,
        )
    if stage == "attn_scale":
        return AttentionScaleTokenLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            window_size=window_size,
            attention_scale=attention_scale,
            lm_head_softcap=lm_head_softcap,
        )
    raise ValueError(f"Unsupported modular-arithmetic stage {stage} for variant={variant}")
