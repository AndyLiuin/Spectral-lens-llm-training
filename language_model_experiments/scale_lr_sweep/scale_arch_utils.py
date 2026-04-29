from __future__ import annotations

from typing import List, Tuple

MODEL_PROFILES = {
    "d12": (12, 6, 768),
    "d24": (24, 16, 1024),
    "d36": (36, 20, 1280),
    "d48": (48, 25, 1600),
}


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def resolve_model_dims(args, default: Tuple[int, int, int] = (12, 6, 768)) -> Tuple[int, int, int]:
    profile = getattr(args, "model_profile", "d12")
    base = MODEL_PROFILES.get(profile, default)
    n_layer = getattr(args, "n_layer", None)
    n_head = getattr(args, "n_head", None)
    n_embd = getattr(args, "n_embd", None)

    layer = base[0] if n_layer is None else int(n_layer)
    head = base[1] if n_head is None else int(n_head)
    embd = base[2] if n_embd is None else int(n_embd)
    return layer, head, embd


def validate_dims(n_layer: int, n_head: int, n_embd: int, require_even_layers: bool = True) -> None:
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    if n_head <= 0:
        raise ValueError(f"n_head must be positive, got {n_head}")
    if n_embd <= 0:
        raise ValueError(f"n_embd must be positive, got {n_embd}")
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    if require_even_layers and (n_layer % 2 != 0):
        raise ValueError(f"n_layer ({n_layer}) must be even for 12-16 architecture family")


def default_skip_attn_layer(n_layer: int) -> int:
    return min(n_layer - 1, n_layer // 2 + 1)


def default_vte_endpoint_k(n_layer: int) -> int:
    return _clamp(int(round(0.25 * n_layer)), 3, max(1, n_layer // 2))


def build_sparse_endpoint_pattern(ve0, ve1, ve2, n_layer: int, endpoint_k: int = 0):
    k = default_vte_endpoint_k(n_layer) if endpoint_k <= 0 else int(endpoint_k)
    k = _clamp(k, 1, max(1, n_layer // 2))
    cycle = [ve0, ve1, ve2]
    front = [cycle[i % len(cycle)] for i in range(k)]
    back = [cycle[i % len(cycle)] for i in range(k)]
    middle = [None] * max(0, n_layer - 2 * k)
    pattern = front + middle + back
    if len(pattern) != n_layer:
        raise ValueError(f"Generated sparse VTE pattern len {len(pattern)} != n_layer {n_layer}")
    return pattern


def auto_layers(n_layer: int, k: int = 5) -> List[int]:
    if n_layer <= 0:
        return []
    if n_layer == 1:
        return [0]
    if k <= 1:
        return [0]

    pts: List[int] = []
    for i in range(k):
        idx = int(round(i * (n_layer - 1) / (k - 1)))
        if idx not in pts:
            pts.append(idx)
    return pts


def parse_layers_spec(spec: str, n_layer: int) -> List[int]:
    if not spec or spec.lower() == "auto":
        return auto_layers(n_layer, k=5)
    if spec.lower() == "all":
        return list(range(n_layer))

    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a_str, b_str = part.split("-", 1)
                a, b = int(a_str), int(b_str)
            except ValueError:
                continue
            lo, hi = sorted((a, b))
            for idx in range(lo, hi + 1):
                if 0 <= idx < n_layer and idx not in out:
                    out.append(idx)
            continue
        try:
            idx = int(part)
        except ValueError:
            continue
        if 0 <= idx < n_layer and idx not in out:
            out.append(idx)
    return sorted(out)


def default_batch_sizes_for_profile(profile: str) -> List[int]:
    table = {
        "d12": [4, 8, 16, 32],
        "d24": [4, 8, 16],
        "d36": [2, 4, 8, 16],
        "d48": [1, 2, 4, 8],
    }
    return table.get(profile, [4, 8, 16, 32])
