from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

EPS = 1e-12


def _stable_seed(seed: int, namespace: str, fixed_samples: bool) -> int:
    payload = f"{int(seed)}::{int(bool(fixed_samples))}::{namespace}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFF_FFFF


@dataclass
class SpectrumMetrics:
    rankme: float
    alpha_head: float
    alpha_tail: float
    top10: float


def _sanitize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    s = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    s = np.maximum(s, 0.0)
    s = np.sort(s)[::-1]
    return s


def normalize_trace(spectrum: np.ndarray) -> np.ndarray:
    s = _sanitize_spectrum(spectrum)
    tr = float(np.sum(s))
    if tr <= EPS:
        return s
    return s / tr


def compute_rankme(spectrum: np.ndarray) -> float:
    s = _sanitize_spectrum(spectrum)
    total = float(np.sum(s))
    if total <= EPS:
        return float("nan")
    p = s / total
    p = np.clip(p, EPS, None)
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_alpha(spectrum: np.ndarray, lo: int, hi: int) -> float:
    s = _sanitize_spectrum(spectrum)
    n = len(s)
    lo = max(1, lo)
    hi = min(max(lo + 2, hi), n)
    if hi - lo < 2:
        return float("nan")
    x = np.arange(lo, hi + 1, dtype=np.float64)
    y = s[lo - 1 : hi]
    good = y > 0
    if good.sum() < 2:
        return float("nan")
    lx = np.log(x[good])
    ly = np.log(y[good])
    slope, _ = np.polyfit(lx, ly, deg=1)
    return float(-slope)


def topk_share(spectrum: np.ndarray, k: int = 10) -> float:
    s = _sanitize_spectrum(spectrum)
    total = float(np.sum(s))
    if total <= EPS:
        return float("nan")
    k = min(k, len(s))
    return float(np.sum(s[:k]) / total)


def covariance_eigenspectrum(matrix: np.ndarray, center: bool = True) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if n <= 1:
        return np.array([], dtype=np.float64)
    # Cov eigenvalues from singular values of centered data matrix.
    svals = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    ev = (svals ** 2) / max(n - 1, 1)
    return np.sort(np.maximum(ev, 0.0))[::-1]


def summarize_spectrum(
    spectrum: np.ndarray,
    alpha_head_range: Tuple[int, int],
    alpha_tail_range: Tuple[int, int],
) -> SpectrumMetrics:
    return SpectrumMetrics(
        rankme=compute_rankme(spectrum),
        alpha_head=compute_alpha(spectrum, *alpha_head_range),
        alpha_tail=compute_alpha(spectrum, *alpha_tail_range),
        top10=topk_share(spectrum, k=10),
    )


def measurement_indices(
    num_examples: int,
    n_samples: int,
    fixed_samples: bool,
    seed: int,
    step: int,
    namespace: str,
) -> np.ndarray:
    n = min(num_examples, n_samples)
    if n <= 0:
        return np.array([], dtype=np.int64)
    base_seed = _stable_seed(seed=seed, namespace=namespace, fixed_samples=fixed_samples)
    if fixed_samples:
        rng = np.random.default_rng(base_seed)
    else:
        rng = np.random.default_rng((base_seed + 9973 * max(step, 1)) & 0xFFFF_FFFF)
    return np.sort(rng.choice(num_examples, size=n, replace=False))


def build_metric_row(
    act_spectrum: np.ndarray,
    grad_spectrum: np.ndarray,
    alpha_head_range: Tuple[int, int],
    alpha_tail_range: Tuple[int, int],
    trace_normalize: bool,
) -> Dict[str, float]:
    a = normalize_trace(act_spectrum) if trace_normalize else _sanitize_spectrum(act_spectrum)
    g = normalize_trace(grad_spectrum) if trace_normalize else _sanitize_spectrum(grad_spectrum)

    act = summarize_spectrum(a, alpha_head_range, alpha_tail_range)
    grad = summarize_spectrum(g, alpha_head_range, alpha_tail_range)

    return {
        "rankme": act.rankme,
        "alpha_head": act.alpha_head,
        "alpha_tail": act.alpha_tail,
        "top10": act.top10,
        "grad_rankme": grad.rankme,
        "grad_alpha_head": grad.alpha_head,
        "grad_alpha_tail": grad.alpha_tail,
        "grad_top10": grad.top10,
    }
