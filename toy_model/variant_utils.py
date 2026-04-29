from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple


def parse_variant_tokens(variant: str) -> set[str]:
    parts = [x.strip().lower() for x in str(variant).split("+") if x.strip()]
    return set(parts) if parts else {"baseline"}


def resolve_variant_settings(
    variant: str,
    *,
    optimizer_name: str = "adamw",
    num_layers: int = 1,
    window_size: int = 0,
    attention_scale: Optional[float] = None,
) -> Dict[str, Any]:
    tokens = parse_variant_tokens(variant)
    needs_unet_depth = any(name in tokens for name in ("unet", "fixed_window", "attn_scale"))
    return {
        "variant": str(variant),
        "optimizer_name": "muon" if "muon" in tokens else optimizer_name,
        "window_size": int(window_size) if any(name in tokens for name in ("fixed_window", "attn_scale")) else 0,
        "attention_scale": attention_scale if "attn_scale" in tokens else None,
        "num_layers": max(int(num_layers), 2) if needs_unet_depth else int(num_layers),
    }


def cumulative_variant_combos(order: List[str]) -> List[Tuple[int, str, str]]:
    combos: List[Tuple[int, str, str]] = []
    active: List[str] = []
    for idx, stage in enumerate(order):
        name = stage.strip().lower()
        if not name:
            continue
        if name != "baseline" and name not in active:
            active.append(name)
        combo = "baseline" if not active else "baseline+" + "+".join(active)
        combos.append((idx, name, combo))
    return combos


def add_optimizer_group_args(parser) -> None:
    parser.add_argument("--embed-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--scalar-lr", type=float, default=None)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-backend", type=str, default="newtonschulz5", choices=["newtonschulz5", "svd"])
    parser.add_argument("--muon-backend-steps", type=int, default=5)
    parser.add_argument("--muon-notebook-lr-defaults", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)


def optimizer_group_kwargs_from_args(args) -> Dict[str, Any]:
    return {
        "embed_lr": getattr(args, "embed_lr", None),
        "head_lr": getattr(args, "head_lr", None),
        "scalar_lr": getattr(args, "scalar_lr", None),
        "muon_lr": getattr(args, "muon_lr", None),
        "muon_momentum": getattr(args, "muon_momentum", 0.95),
        "muon_backend": getattr(args, "muon_backend", "newtonschulz5"),
        "muon_backend_steps": getattr(args, "muon_backend_steps", 5),
        "muon_notebook_lr_defaults": getattr(args, "muon_notebook_lr_defaults", True),
        "weight_decay": getattr(args, "weight_decay", 0.0),
    }
