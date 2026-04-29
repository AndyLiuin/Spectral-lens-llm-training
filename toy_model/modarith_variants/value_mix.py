from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import ScaleStyleBlock, apply_norm


class ScaleStyleTokenLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        *,
        use_unet: bool = False,
        window_size: int = 0,
        attention_scale: Optional[float] = None,
        lm_head_softcap: float = 30.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.use_unet = bool(use_unet)
        self.lm_head_softcap = float(lm_head_softcap)

        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                ScaleStyleBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    use_rope=True,
                    use_value_mix=True,
                    use_x0_mix=True,
                    window_size=window_size,
                    attention_scale=attention_scale,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_unet:
            dec_layers = self.n_layers // 2
            self.skip_weights = nn.Parameter(torch.ones(dec_layers))
        else:
            self.skip_weights = None

        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        v1 = None

        if not self.use_unet:
            for block in self.blocks:
                x, v1 = block(x, v1, x0)
            return x

        enc = self.n_layers // 2
        dec = self.n_layers - enc
        skips = []

        for i in range(enc):
            x, v1 = self.blocks[i](x, v1, x0)
            skips.append(x)

        for i in range(dec):
            x = x + self.skip_weights[i] * skips.pop()
            x, v1 = self.blocks[enc + i](x, v1, x0)
        return x

    def forward(self, x_tokens: torch.Tensor, return_repr: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_embed(x_tokens)
        x = apply_norm(x)
        h = apply_norm(self._forward_blocks(x))
        logits = self.lm_head(h)
        if self.lm_head_softcap > 0.0:
            sc = self.lm_head_softcap
            logits = sc * torch.tanh(logits / sc)
        if return_repr:
            return logits, h
        return logits, None


class ValueMixTokenLM(ScaleStyleTokenLM):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        lm_head_softcap: float = 30.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            use_unet=False,
            window_size=0,
            attention_scale=None,
            lm_head_softcap=lm_head_softcap,
        )
