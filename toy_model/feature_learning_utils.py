from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-30


@dataclass
class ProbeSplit:
    x: np.ndarray
    y: np.ndarray
    state_labels: np.ndarray
    metadata: Dict[str, object]


@dataclass
class FeatureProbeOutput:
    summary_row: Dict[str, object]
    causal_row: Dict[str, object]
    pca_rows: List[Dict[str, object]]


def _factor_list(n: int) -> list[int]:
    out: list[int] = []
    for p in range(2, int(np.sqrt(n)) + 1):
        if n % p == 0:
            out.append(p)
    return out


def _sample_step(rng: np.random.Generator, c: int, min_step_frac: float, allow_noncoprime: bool, noncoprime_prob: float) -> int:
    d_min = max(1, int(min_step_frac * c))
    if d_min >= c:
        d = 1
    else:
        d = int(rng.integers(d_min, c))

    if allow_noncoprime and rng.random() < noncoprime_prob:
        factors = _factor_list(c)
        if factors:
            g = int(rng.choice(np.asarray(factors, dtype=np.int64)))
            d = (d // g) * g
            d = max(g, min(d, c - 1))
    return d


def _sample_component_weight(rng: np.random.Generator, pareto_alpha: float) -> float:
    if pareto_alpha > 0.0:
        return float(1.0 + rng.pareto(pareto_alpha))
    return 1.0


def _sample_nuisance_component(
    rng: np.random.Generator,
    seq_k: np.ndarray,
    vocab_size: int,
    cs: np.ndarray,
    c_probs: np.ndarray,
    zipf_o: float,
    min_step_frac: float,
    allow_noncoprime: bool,
    noncoprime_prob: float,
    pareto_alpha: float,
) -> tuple[np.ndarray, int, int, int, int, float]:
    c = int(rng.choice(cs, p=c_probs))
    max_o = vocab_size - c
    if max_o <= 0:
        o = 0
    else:
        os = np.arange(0, max_o + 1, dtype=np.int64)
        o_w = np.power(os.astype(np.float64) + 1.0, -float(zipf_o))
        o_probs = o_w / (np.sum(o_w) + 1e-12)
        o = int(rng.choice(os, p=o_probs))
    d = _sample_step(rng, c, min_step_frac, allow_noncoprime, noncoprime_prob)
    a = int(rng.integers(0, c))
    s = (a + d * seq_k) % c
    t = (o + s).astype(np.float64)
    w = _sample_component_weight(rng, pareto_alpha)
    return t, c, o, d, a, w


def build_fixed_band_probe_split(
    n: int,
    seq_len: int,
    vocab_size: int,
    c0: int,
    o0: int,
    min_step_frac: float = 0.125,
    allow_noncoprime: bool = True,
    noncoprime_prob: float = 0.3,
    seed: int = 0,
) -> ProbeSplit:
    if c0 <= 1:
        raise ValueError("c0 must be > 1.")
    if o0 < 0 or o0 + c0 > vocab_size:
        raise ValueError("Band [o0, o0+c0) must lie within vocab.")

    rng = np.random.default_rng(seed)
    k = np.arange(seq_len + 1, dtype=np.int64)
    x = np.zeros((n, seq_len), dtype=np.int64)
    y = np.zeros((n, seq_len), dtype=np.int64)
    state_labels = np.zeros((n, seq_len), dtype=np.int64)

    for i in range(n):
        d = _sample_step(rng, c0, min_step_frac, allow_noncoprime, noncoprime_prob)
        a = int(rng.integers(0, c0))
        s = (a + d * k) % c0
        t = (o0 + s).astype(np.int64)
        x[i, :] = t[:-1]
        y[i, :] = t[1:]
        state_labels[i, :] = s[:-1]

    return ProbeSplit(
        x=x,
        y=y,
        state_labels=state_labels,
        metadata={
            "probe_type": "clean_band",
            "anchor_band": f"{c0}:{o0}",
            "matches_training_distribution": False,
            "anchored_component_forced": True,
        },
    )


def build_matched_band_probe_split(
    n: int,
    seq_len: int,
    vocab_size: int,
    c0: int,
    o0: int,
    zipf_c: float,
    zipf_o: float,
    c_min: int,
    min_step_frac: float = 0.125,
    allow_noncoprime: bool = True,
    noncoprime_prob: float = 0.3,
    mix_components_min: int = 1,
    mix_components_max: int = 1,
    component_weight_pareto_alpha: float = 0.0,
    token_noise_std: float = 0.0,
    token_noise_t_df: float = 0.0,
    seed: int = 0,
) -> ProbeSplit:
    if c0 <= 1:
        raise ValueError("c0 must be > 1.")
    if o0 < 0 or o0 + c0 > vocab_size:
        raise ValueError("Band [o0, o0+c0) must lie within vocab.")

    rng = np.random.default_rng(seed)
    k = np.arange(seq_len + 1, dtype=np.int64)
    x = np.zeros((n, seq_len), dtype=np.int64)
    y = np.zeros((n, seq_len), dtype=np.int64)
    state_labels = np.zeros((n, seq_len), dtype=np.int64)

    cs = np.arange(c_min, vocab_size + 1, dtype=np.int64)
    c_w = np.power(cs.astype(np.float64), -float(zipf_c))
    c_probs = c_w / (np.sum(c_w) + 1e-12)
    m_lo = max(1, int(mix_components_min))
    m_hi = max(m_lo, int(mix_components_max))
    component_counts: List[int] = []
    anchor_weights: List[float] = []

    for i in range(n):
        total_components = int(rng.integers(m_lo, m_hi + 1))
        component_counts.append(total_components)

        d0 = _sample_step(rng, c0, min_step_frac, allow_noncoprime, noncoprime_prob)
        a0 = int(rng.integers(0, c0))
        s0 = (a0 + d0 * k) % c0
        t_mix = np.zeros(seq_len + 1, dtype=np.float64)
        w_sum = 0.0

        w0 = _sample_component_weight(rng, component_weight_pareto_alpha)
        t_mix += w0 * (o0 + s0).astype(np.float64)
        w_sum += w0
        anchor_weights.append(w0)

        for _ in range(max(0, total_components - 1)):
            t, _, _, _, _, w = _sample_nuisance_component(
                rng=rng,
                seq_k=k,
                vocab_size=vocab_size,
                cs=cs,
                c_probs=c_probs,
                zipf_o=zipf_o,
                min_step_frac=min_step_frac,
                allow_noncoprime=allow_noncoprime,
                noncoprime_prob=noncoprime_prob,
                pareto_alpha=component_weight_pareto_alpha,
            )
            t_mix += w * t
            w_sum += w

        if w_sum > 0:
            t_mix /= w_sum

        if token_noise_std > 0.0:
            if token_noise_t_df > 2.0:
                noise = rng.standard_t(df=token_noise_t_df, size=(seq_len + 1,))
                noise = noise * np.sqrt((token_noise_t_df - 2.0) / token_noise_t_df)
            else:
                noise = rng.normal(loc=0.0, scale=1.0, size=(seq_len + 1,))
            t_mix = t_mix + float(token_noise_std) * noise.astype(np.float64)

        t_idx = np.clip(np.rint(t_mix), 0, vocab_size - 1).astype(np.int64)
        x[i, :] = t_idx[:-1]
        y[i, :] = t_idx[1:]
        state_labels[i, :] = s0[:-1]

    return ProbeSplit(
        x=x,
        y=y,
        state_labels=state_labels,
        metadata={
            "probe_type": "matched_band",
            "anchor_band": f"{c0}:{o0}",
            "matches_training_distribution": True,
            "anchored_component_forced": True,
            "mean_total_components": float(np.mean(component_counts)) if component_counts else 1.0,
            "mean_anchor_weight": float(np.mean(anchor_weights)) if anchor_weights else 1.0,
        },
    )


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= EPS:
        return np.zeros_like(x, dtype=np.float64)
    return x / s


def fourier_power_over_token_axis(X_band: np.ndarray, center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X_band, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    Fk = np.fft.fft(X, axis=0)
    P = (np.abs(Fk) ** 2).sum(axis=1)
    Pn = normalize_distribution(P)
    return P, Pn


def fourier_energy_1d(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"Expected 2D H, got shape {H.shape}")
    Fm = np.fft.fft(H, axis=0)
    E = (np.abs(Fm) ** 2).sum(axis=1)
    En = normalize_distribution(E)
    return E, En


def positive_freq_indices(c0: int) -> np.ndarray:
    half = c0 // 2
    if half < 1:
        return np.array([], dtype=np.int64)
    return np.arange(1, half + 1, dtype=np.int64)


def positive_freq_slice(Pn: np.ndarray) -> np.ndarray:
    Pn = np.asarray(Pn, dtype=np.float64)
    idx = positive_freq_indices(len(Pn))
    return Pn[idx] if idx.size > 0 else np.array([], dtype=np.float64)


def fourier_peak(Pn: np.ndarray) -> float:
    pos = positive_freq_slice(Pn)
    if pos.size == 0:
        return 0.0
    return float(np.max(pos))


def dominant_positive_freq(Pn: np.ndarray) -> Optional[int]:
    pos = positive_freq_slice(Pn)
    if pos.size == 0:
        return None
    return int(np.argmax(pos) + 1)


def gini(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, np.inf)
    s = float(x.sum())
    if s < eps:
        return 0.0
    x = np.sort(x / s)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * x).sum() / n) - (n + 1.0) / n)


def canonicalize_freqs(freqs_pos: Sequence[int], c0: int, keep_dc: bool = True) -> List[int]:
    keep = set([0]) if keep_dc else set()
    for f in freqs_pos:
        f = int(f) % c0
        keep.add(f)
        keep.add((-f) % c0)
    return sorted(keep)


def fft_project_band(X_band: np.ndarray, keep_freqs_full: Sequence[int], mode: str = "keep", center: bool = True) -> np.ndarray:
    X = np.asarray(X_band, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X_band, got shape {X.shape}")
    if center:
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
    else:
        mu = np.zeros((1, X.shape[1]), dtype=np.float64)
        Xc = X

    Fk = np.fft.fft(Xc, axis=0)
    mask = np.zeros((X.shape[0],), dtype=np.float64)
    keep_set = set(int(f) % X.shape[0] for f in keep_freqs_full)
    if mode == "keep":
        for f in keep_set:
            mask[f] = 1.0
    elif mode == "drop":
        mask[:] = 1.0
        for f in keep_set:
            mask[f] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    Fp = Fk * mask[:, None]
    Xp = np.fft.ifft(Fp, axis=0).real + mu
    return Xp.astype(X_band.dtype, copy=False)


def choose_control_freqs(c0: int, key_freqs_pos: Sequence[int], n_ctrl: int = 1, seed: int = 0) -> List[int]:
    idx = positive_freq_indices(c0).tolist()
    key = {int(f) for f in key_freqs_pos}
    pool = [f for f in idx if f not in key]
    if not pool:
        return []
    n_ctrl = max(1, min(int(n_ctrl), len(pool)))
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.asarray(pool, dtype=np.int64), size=n_ctrl, replace=False)
    return [int(x) for x in np.asarray(pick).tolist()]


@contextlib.contextmanager
def patch_token_embedding_band(model: torch.nn.Module, o0: int, c0: int, new_band: np.ndarray):
    if not hasattr(model, "tok_embed"):
        raise AttributeError("Model has no tok_embed; only token-LM Track A models are supported.")
    weight: torch.Tensor = model.tok_embed.weight
    device = weight.device
    dtype = weight.dtype
    band = torch.as_tensor(new_band, device=device, dtype=dtype)
    if band.shape != (c0, weight.shape[1]):
        raise ValueError(f"Expected band shape {(c0, weight.shape[1])}, got {tuple(band.shape)}")
    with torch.no_grad():
        old = weight[o0 : o0 + c0].detach().clone()
        weight[o0 : o0 + c0] = band
    try:
        yield
    finally:
        with torch.no_grad():
            weight[o0 : o0 + c0] = old


@torch.no_grad()
def evaluate_lm_loss(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    vocab_size: int,
    batch_size: int = 256,
) -> float:
    model.eval()
    losses: List[float] = []
    for start in range(0, len(x), batch_size):
        end = min(len(x), start + batch_size)
        xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
        yb = torch.as_tensor(y[start:end], dtype=torch.long, device=device)
        logits, _ = model(xb, return_repr=True)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="mean")
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def compute_conditional_mean_h(
    model: torch.nn.Module,
    x_tokens: np.ndarray,
    c0: int,
    o0: int,
    device: str,
    pos: int = -1,
    batch_size: int = 256,
    center_over_m: bool = True,
    state_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if not hasattr(model, "tok_embed"):
        raise AttributeError("Model has no tok_embed; expected TransformerTokenLM.")
    d = int(model.tok_embed.weight.shape[1])
    sums = np.zeros((c0, d), dtype=np.float64)
    counts = np.zeros((c0,), dtype=np.int64)

    for start in range(0, len(x_tokens), batch_size):
        end = min(len(x_tokens), start + batch_size)
        xb = torch.as_tensor(x_tokens[start:end], dtype=torch.long, device=device)
        _, h = model(xb, return_repr=True)
        if h is None:
            raise RuntimeError("Model did not return representation tensor.")
        p = int(pos)
        if p < 0:
            p = h.shape[1] + p
        p = max(0, min(p, h.shape[1] - 1))

        rep = h[:, p, :]
        if state_labels is None:
            tok = xb[:, p]
            m = (tok - int(o0)).clamp(min=0, max=c0 - 1).to(torch.int64).cpu().numpy()
        else:
            m = np.asarray(state_labels[start:end, p], dtype=np.int64)
            m = np.clip(m, 0, c0 - 1)
        rep_np = rep.cpu().numpy()
        for mm in np.unique(m):
            mask = m == mm
            counts[mm] += int(mask.sum())
            sums[mm] += rep_np[mask].sum(axis=0)

    H = sums / np.maximum(counts[:, None], 1)
    if center_over_m:
        H = H - H.mean(axis=0, keepdims=True)
    return H.astype(np.float64), counts


def neuron_logit_map_last_block(model: torch.nn.Module) -> Optional[np.ndarray]:
    if not hasattr(model, "blocks") or not hasattr(model, "lm_head"):
        return None
    if len(model.blocks) == 0:
        return None
    last_block = model.blocks[-1]
    if not hasattr(last_block, "ff"):
        return None

    ff = last_block.ff
    w2 = None
    if hasattr(ff, "__len__") and len(ff) >= 3 and isinstance(ff[-1], torch.nn.Linear):
        w2 = ff[-1].weight.detach()
    else:
        linear_layers = [m for m in ff.modules() if isinstance(m, torch.nn.Linear)]
        if linear_layers:
            w2 = linear_layers[-1].weight.detach()
    if w2 is None:
        return None

    wu = model.lm_head.weight.detach()
    w2_in = w2.T
    e_in = (w2_in @ wu.T).cpu().numpy()
    return e_in.T


@torch.no_grad()
def embedding_band_fourier(model: torch.nn.Module, c0: int, o0: int) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "tok_embed"):
        return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    we = model.tok_embed.weight.detach().cpu().numpy()
    X = we[o0 : o0 + c0, :]
    _, pn = fourier_power_over_token_axis(X, center=True)
    return pn, X


@torch.no_grad()
def neuronlogit_band_fourier(model: torch.nn.Module, c0: int, o0: int) -> Tuple[np.ndarray, np.ndarray]:
    E = neuron_logit_map_last_block(model)
    if E is None:
        return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    X = E[o0 : o0 + c0, :]
    _, pn = fourier_power_over_token_axis(X, center=True)
    return pn, X


def pca_h_over_m(H: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=np.float64)
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = max(1, min(int(k), U.shape[1]))
    return U[:, :k], S[:k], Vt[:k, :]


def pc_positive_spectrum(u_col: np.ndarray) -> np.ndarray:
    u = np.asarray(u_col, dtype=np.float64)
    u = u - u.mean()
    Fv = np.fft.fft(u)
    P = normalize_distribution(np.abs(Fv) ** 2)
    return positive_freq_slice(P)


def pca_frequency_summary(H: np.ndarray, pca_k: int = 8) -> Dict[str, object]:
    U, S, Vt = pca_h_over_m(H, k=pca_k)
    s2 = S**2
    evr = s2 / (s2.sum() + EPS)

    peak_mass: List[float] = []
    dominant_freq: List[int] = []
    pos_specs: List[np.ndarray] = []
    for j in range(U.shape[1]):
        pos = pc_positive_spectrum(U[:, j])
        pos_specs.append(pos)
        if pos.size == 0:
            peak_mass.append(float("nan"))
            dominant_freq.append(-1)
        else:
            peak_mass.append(float(np.max(pos)))
            dominant_freq.append(int(np.argmax(pos) + 1))

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "evr": evr,
        "peak_mass": np.asarray(peak_mass, dtype=np.float64),
        "dominant_freq": np.asarray(dominant_freq, dtype=np.int64),
        "pos_specs": pos_specs,
    }


def _hidden_pos_index(h: torch.Tensor, pos: int) -> int:
    p = int(pos)
    if p < 0:
        p = h.shape[1] + p
    return max(0, min(p, h.shape[1] - 1))


@torch.no_grad()
def _forward_hidden_states(model: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "tok_embed") or not hasattr(model, "lm_head"):
        raise AttributeError("Expected token-LM model with tok_embed and lm_head.")
    _, h = model(xb, return_repr=True)
    if h is None:
        raise RuntimeError("Model did not return representation tensor.")
    return h


def _logits_from_repr(model: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(h)
    softcap = float(getattr(model, "lm_head_softcap", 0.0) or 0.0)
    if softcap > 0.0:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _projector_from_rows(rows: np.ndarray, d_model: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((d_model, d_model), dtype=np.float64)
    q, _ = np.linalg.qr(rows.T)
    if q.ndim == 1:
        q = q[:, None]
    return q @ q.T


@torch.no_grad()
def evaluate_lm_loss_with_hidden_projection(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    vocab_size: int,
    projector: np.ndarray,
    mean_vec: np.ndarray,
    pos: int = -1,
    mode: str = "keep",
    batch_size: int = 256,
) -> float:
    model.eval()
    losses: List[float] = []
    proj_t = None
    mean_t = None
    for start in range(0, len(x), batch_size):
        end = min(len(x), start + batch_size)
        xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
        yb = torch.as_tensor(y[start:end], dtype=torch.long, device=device)
        h = _forward_hidden_states(model, xb)
        p = _hidden_pos_index(h, pos)
        rep = h[:, p, :]
        if proj_t is None:
            proj_t = torch.as_tensor(projector, dtype=rep.dtype, device=rep.device)
            mean_t = torch.as_tensor(mean_vec, dtype=rep.dtype, device=rep.device)
        centered = rep - mean_t
        if mode == "keep":
            edited = mean_t + centered @ proj_t
        elif mode == "drop":
            eye = torch.eye(proj_t.shape[0], device=proj_t.device, dtype=proj_t.dtype)
            edited = mean_t + centered @ (eye - proj_t)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        h = h.clone()
        h[:, p, :] = edited
        logits = _logits_from_repr(model, h)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="mean")
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_hidden_last_token(
    model: torch.nn.Module,
    x: np.ndarray,
    device: str,
    pos: int = -1,
    batch_size: int = 256,
) -> np.ndarray:
    reps: List[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        end = min(len(x), start + batch_size)
        xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
        h = _forward_hidden_states(model, xb)
        p = _hidden_pos_index(h, pos)
        reps.append(h[:, p, :].detach().cpu().numpy())
    return np.concatenate(reps, axis=0) if reps else np.empty((0, 0), dtype=np.float64)


def _select_key_and_control_subspaces(
    pca_summary: Dict[str, object],
    key_freq: Optional[int],
) -> tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    Vt = np.asarray(pca_summary["Vt"], dtype=np.float64)
    dom = np.asarray(pca_summary["dominant_freq"], dtype=np.int64)
    peak = np.asarray(pca_summary["peak_mass"], dtype=np.float64)
    order = np.argsort(np.nan_to_num(peak, nan=-1.0))[::-1]

    key_idx = [int(i) for i in order if key_freq is not None and int(dom[i]) == int(key_freq)]
    if not key_idx and len(order) > 0:
        key_idx = [int(order[0])]
    ctrl_idx = [int(i) for i in order if i not in key_idx and int(dom[i]) > 0]
    ctrl_idx = ctrl_idx[: len(key_idx)] if key_idx else []

    key_rows = Vt[key_idx, :] if key_idx else np.empty((0, Vt.shape[1]), dtype=np.float64)
    ctrl_rows = Vt[ctrl_idx, :] if ctrl_idx else np.empty((0, Vt.shape[1]), dtype=np.float64)
    return key_rows, ctrl_rows, key_idx, ctrl_idx


@torch.no_grad()
def run_feature_probe_once(
    model: torch.nn.Module,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    state_labels: np.ndarray,
    c0: int,
    o0: int,
    device: str,
    vocab_size: int,
    pos: int = -1,
    pca_k: int = 8,
    control_seed: int = 0,
    batch_size: int = 256,
) -> FeatureProbeOutput:
    base_loss = evaluate_lm_loss(model=model, x=probe_x, y=probe_y, device=device, vocab_size=vocab_size, batch_size=batch_size)

    H, counts = compute_conditional_mean_h(
        model=model,
        x_tokens=probe_x,
        c0=c0,
        o0=o0,
        device=device,
        pos=pos,
        batch_size=batch_size,
        center_over_m=True,
        state_labels=state_labels,
    )
    _, Hn = fourier_energy_1d(H)
    H_peak = fourier_peak(Hn)
    H_gini = gini(Hn)
    key_freq = dominant_positive_freq(Hn)

    emb_pn, _ = embedding_band_fourier(model=model, c0=c0, o0=o0)
    e_pn, _ = neuronlogit_band_fourier(model=model, c0=c0, o0=o0)

    emb_peak = fourier_peak(emb_pn) if emb_pn.size else float("nan")
    emb_gini = gini(emb_pn) if emb_pn.size else float("nan")
    e_peak = fourier_peak(e_pn) if e_pn.size else float("nan")
    e_gini = gini(e_pn) if e_pn.size else float("nan")
    H_mass_total = float(np.sum(Hn)) if Hn.size else float("nan")
    E_mass_total = float(np.sum(e_pn)) if e_pn.size else float("nan")
    Emb_mass_total = float(np.sum(emb_pn)) if emb_pn.size else float("nan")
    H_dc_mass = float(Hn[0]) if Hn.size else float("nan")
    H_pos_mass = float(np.sum(positive_freq_slice(Hn))) if Hn.size else float("nan")

    embedding_keep_key_loss = np.nan
    embedding_drop_key_loss = np.nan
    embedding_keep_ctrl_loss = np.nan
    embedding_drop_ctrl_loss = np.nan
    ctrl_freq = None

    if key_freq is not None:
        key_pos = [int(key_freq)]
        keep_full = canonicalize_freqs(key_pos, c0=c0, keep_dc=True)
        ctrl_pos = choose_control_freqs(c0=c0, key_freqs_pos=key_pos, n_ctrl=len(key_pos), seed=control_seed)
        ctrl_freq = int(ctrl_pos[0]) if ctrl_pos else None
        ctrl_full = canonicalize_freqs(ctrl_pos, c0=c0, keep_dc=True) if ctrl_pos else []

        we = model.tok_embed.weight.detach().cpu().numpy()
        band = we[o0 : o0 + c0, :].astype(np.float64, copy=True)
        band_keep = fft_project_band(band, keep_full, mode="keep", center=True)
        band_drop = fft_project_band(band, keep_full, mode="drop", center=True)
        with patch_token_embedding_band(model=model, o0=o0, c0=c0, new_band=band_keep):
            embedding_keep_key_loss = evaluate_lm_loss(model=model, x=probe_x, y=probe_y, device=device, vocab_size=vocab_size, batch_size=batch_size)
        with patch_token_embedding_band(model=model, o0=o0, c0=c0, new_band=band_drop):
            embedding_drop_key_loss = evaluate_lm_loss(model=model, x=probe_x, y=probe_y, device=device, vocab_size=vocab_size, batch_size=batch_size)

        if ctrl_full:
            band_keep_c = fft_project_band(band, ctrl_full, mode="keep", center=True)
            band_drop_c = fft_project_band(band, ctrl_full, mode="drop", center=True)
            with patch_token_embedding_band(model=model, o0=o0, c0=c0, new_band=band_keep_c):
                embedding_keep_ctrl_loss = evaluate_lm_loss(model=model, x=probe_x, y=probe_y, device=device, vocab_size=vocab_size, batch_size=batch_size)
            with patch_token_embedding_band(model=model, o0=o0, c0=c0, new_band=band_drop_c):
                embedding_drop_ctrl_loss = evaluate_lm_loss(model=model, x=probe_x, y=probe_y, device=device, vocab_size=vocab_size, batch_size=batch_size)

    pca = pca_frequency_summary(H=H, pca_k=pca_k)
    pca_rows: List[Dict[str, object]] = []
    for i in range(len(pca["evr"])):
        pca_rows.append(
            {
                "pc_index": int(i + 1),
                "pca_evr": float(pca["evr"][i]),
                "pca_peak_mass": float(pca["peak_mass"][i]),
                "dominant_freq": int(pca["dominant_freq"][i]),
            }
        )

    hidden_keep_key_loss = np.nan
    hidden_drop_key_loss = np.nan
    hidden_keep_ctrl_loss = np.nan
    hidden_drop_ctrl_loss = np.nan
    hidden_key_rank = 0
    hidden_ctrl_rank = 0

    hidden_reps = collect_hidden_last_token(model=model, x=probe_x, device=device, pos=pos, batch_size=batch_size)
    mean_vec = hidden_reps.mean(axis=0) if hidden_reps.size else np.array([], dtype=np.float64)
    key_rows, ctrl_rows, key_idx, ctrl_idx = _select_key_and_control_subspaces(pca_summary=pca, key_freq=key_freq)
    hidden_key_rank = int(len(key_idx))
    hidden_ctrl_rank = int(len(ctrl_idx))

    if mean_vec.size and hidden_key_rank > 0:
        key_proj = _projector_from_rows(key_rows, mean_vec.shape[0])
        hidden_keep_key_loss = evaluate_lm_loss_with_hidden_projection(
            model=model,
            x=probe_x,
            y=probe_y,
            device=device,
            vocab_size=vocab_size,
            projector=key_proj,
            mean_vec=mean_vec,
            pos=pos,
            mode="keep",
            batch_size=batch_size,
        )
        hidden_drop_key_loss = evaluate_lm_loss_with_hidden_projection(
            model=model,
            x=probe_x,
            y=probe_y,
            device=device,
            vocab_size=vocab_size,
            projector=key_proj,
            mean_vec=mean_vec,
            pos=pos,
            mode="drop",
            batch_size=batch_size,
        )
    if mean_vec.size and hidden_ctrl_rank > 0:
        ctrl_proj = _projector_from_rows(ctrl_rows, mean_vec.shape[0])
        hidden_keep_ctrl_loss = evaluate_lm_loss_with_hidden_projection(
            model=model,
            x=probe_x,
            y=probe_y,
            device=device,
            vocab_size=vocab_size,
            projector=ctrl_proj,
            mean_vec=mean_vec,
            pos=pos,
            mode="keep",
            batch_size=batch_size,
        )
        hidden_drop_ctrl_loss = evaluate_lm_loss_with_hidden_projection(
            model=model,
            x=probe_x,
            y=probe_y,
            device=device,
            vocab_size=vocab_size,
            projector=ctrl_proj,
            mean_vec=mean_vec,
            pos=pos,
            mode="drop",
            batch_size=batch_size,
        )

    summary_row = {
        "H_peak": float(H_peak),
        "H_gini": float(H_gini),
        "E_peak": float(e_peak),
        "E_gini": float(e_gini),
        "Emb_peak": float(emb_peak),
        "Emb_gini": float(emb_gini),
        "H_mass_total": H_mass_total,
        "E_mass_total": E_mass_total,
        "Emb_mass_total": Emb_mass_total,
        "H_dc_mass": H_dc_mass,
        "H_pos_mass": H_pos_mass,
        "dominant_freq": int(key_freq) if key_freq is not None else -1,
        "pca_peak_mass_pc1": float(pca_rows[0]["pca_peak_mass"]) if pca_rows else float("nan"),
        "base_probe_loss": float(base_loss),
        "probe_support_min": int(np.min(counts)) if counts.size else 0,
        "probe_support_mean": float(np.mean(counts)) if counts.size else 0.0,
        "hidden_key_rank": hidden_key_rank,
        "hidden_ctrl_rank": hidden_ctrl_rank,
    }

    causal_row = {
        "keep_key_loss": float(embedding_keep_key_loss),
        "drop_key_loss": float(embedding_drop_key_loss),
        "keep_ctrl_loss": float(embedding_keep_ctrl_loss),
        "drop_ctrl_loss": float(embedding_drop_ctrl_loss),
        "delta_keep": float(base_loss - embedding_keep_key_loss) if np.isfinite(embedding_keep_key_loss) else float("nan"),
        "delta_drop": float(base_loss - embedding_drop_key_loss) if np.isfinite(embedding_drop_key_loss) else float("nan"),
        "delta_keep_ctrl": float(base_loss - embedding_keep_ctrl_loss) if np.isfinite(embedding_keep_ctrl_loss) else float("nan"),
        "delta_drop_ctrl": float(base_loss - embedding_drop_ctrl_loss) if np.isfinite(embedding_drop_ctrl_loss) else float("nan"),
        "delta_keep_vs_ctrl": (
            float((base_loss - embedding_keep_key_loss) - (base_loss - embedding_keep_ctrl_loss))
            if np.isfinite(embedding_keep_key_loss) and np.isfinite(embedding_keep_ctrl_loss)
            else float("nan")
        ),
        "delta_drop_vs_ctrl": (
            float((base_loss - embedding_drop_ctrl_loss) - (base_loss - embedding_drop_key_loss))
            if np.isfinite(embedding_drop_key_loss) and np.isfinite(embedding_drop_ctrl_loss)
            else float("nan")
        ),
        "keep_key_minus_base": float(embedding_keep_key_loss - base_loss) if np.isfinite(embedding_keep_key_loss) else float("nan"),
        "drop_key_minus_base": float(embedding_drop_key_loss - base_loss) if np.isfinite(embedding_drop_key_loss) else float("nan"),
        "keep_ctrl_minus_base": float(embedding_keep_ctrl_loss - base_loss) if np.isfinite(embedding_keep_ctrl_loss) else float("nan"),
        "drop_ctrl_minus_base": float(embedding_drop_ctrl_loss - base_loss) if np.isfinite(embedding_drop_ctrl_loss) else float("nan"),
        "key_freq": int(key_freq) if key_freq is not None else -1,
        "ctrl_freq": int(ctrl_freq) if ctrl_freq is not None else -1,
        "embedding_keep_key_loss": float(embedding_keep_key_loss),
        "embedding_drop_key_loss": float(embedding_drop_key_loss),
        "embedding_keep_ctrl_loss": float(embedding_keep_ctrl_loss),
        "embedding_drop_ctrl_loss": float(embedding_drop_ctrl_loss),
        "hidden_keep_key_loss": float(hidden_keep_key_loss),
        "hidden_drop_key_loss": float(hidden_drop_key_loss),
        "hidden_keep_ctrl_loss": float(hidden_keep_ctrl_loss),
        "hidden_drop_ctrl_loss": float(hidden_drop_ctrl_loss),
        "hidden_delta_keep": float(base_loss - hidden_keep_key_loss) if np.isfinite(hidden_keep_key_loss) else float("nan"),
        "hidden_delta_drop": float(base_loss - hidden_drop_key_loss) if np.isfinite(hidden_drop_key_loss) else float("nan"),
        "hidden_delta_keep_ctrl": float(base_loss - hidden_keep_ctrl_loss) if np.isfinite(hidden_keep_ctrl_loss) else float("nan"),
        "hidden_delta_drop_ctrl": float(base_loss - hidden_drop_ctrl_loss) if np.isfinite(hidden_drop_ctrl_loss) else float("nan"),
        "hidden_delta_keep_vs_ctrl": (
            float((base_loss - hidden_keep_key_loss) - (base_loss - hidden_keep_ctrl_loss))
            if np.isfinite(hidden_keep_key_loss) and np.isfinite(hidden_keep_ctrl_loss)
            else float("nan")
        ),
        "hidden_delta_drop_vs_ctrl": (
            float((base_loss - hidden_drop_ctrl_loss) - (base_loss - hidden_drop_key_loss))
            if np.isfinite(hidden_drop_key_loss) and np.isfinite(hidden_drop_ctrl_loss)
            else float("nan")
        ),
        "hidden_key_rank": hidden_key_rank,
        "hidden_ctrl_rank": hidden_ctrl_rank,
    }

    return FeatureProbeOutput(summary_row=summary_row, causal_row=causal_row, pca_rows=pca_rows)
