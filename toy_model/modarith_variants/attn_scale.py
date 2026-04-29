from __future__ import annotations

from typing import Optional

from .value_mix import ScaleStyleTokenLM


class AttentionScaleTokenLM(ScaleStyleTokenLM):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        window_size: int = 8,
        attention_scale: Optional[float] = None,
        lm_head_softcap: float = 30.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=max(int(n_layers), 2),
            ff_mult=ff_mult,
            use_unet=True,
            window_size=max(1, int(window_size)),
            attention_scale=attention_scale,
            lm_head_softcap=lm_head_softcap,
        )
