from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class ModArithDatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: Dict[str, object]
    latents_train: Optional[Dict[str, np.ndarray]] = None
    latents_val: Optional[Dict[str, np.ndarray]] = None
    latents_test: Optional[Dict[str, np.ndarray]] = None


def _factor_list(n: int) -> list[int]:
    out: list[int] = []
    for p in range(2, int(np.sqrt(n)) + 1):
        if n % p == 0:
            out.append(p)
    return out


def _sample_split(
    n: int,
    seq_len: int,
    vocab_size: int,
    zipf_c: float,
    zipf_o: float,
    c_min: int,
    min_step_frac: float,
    allow_noncoprime: bool,
    noncoprime_prob: float,
    mix_components_min: int,
    mix_components_max: int,
    component_weight_pareto_alpha: float,
    token_noise_std: float,
    token_noise_t_df: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    cs = np.arange(c_min, vocab_size + 1, dtype=np.int64)
    c_w = np.power(cs.astype(np.float64), -float(zipf_c))
    c_probs = c_w / (np.sum(c_w) + 1e-12)
    m_lo = max(1, int(mix_components_min))
    m_hi = max(m_lo, int(mix_components_max))

    x = np.zeros((n, seq_len), dtype=np.int64)
    y = np.zeros((n, seq_len), dtype=np.int64)
    latents = {
        "component_count": np.zeros((n,), dtype=np.int64),
        "c_vals": -np.ones((n, m_hi), dtype=np.int64),
        "o_vals": -np.ones((n, m_hi), dtype=np.int64),
        "d_vals": -np.ones((n, m_hi), dtype=np.int64),
        "a_vals": -np.ones((n, m_hi), dtype=np.int64),
        "weights": np.zeros((n, m_hi), dtype=np.float64),
    }

    k = np.arange(seq_len + 1, dtype=np.int64)

    for i in range(n):
        m = int(rng.integers(m_lo, m_hi + 1))
        latents["component_count"][i] = m

        t_mix = np.zeros(seq_len + 1, dtype=np.float64)
        w_sum = 0.0

        for j in range(m):
            c = int(rng.choice(cs, p=c_probs))

            max_o = vocab_size - c
            if max_o <= 0:
                o = 0
            else:
                os = np.arange(0, max_o + 1, dtype=np.int64)
                o_w = np.power(os.astype(np.float64) + 1.0, -float(zipf_o))
                o_probs = o_w / (np.sum(o_w) + 1e-12)
                o = int(rng.choice(os, p=o_probs))

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

            a = int(rng.integers(0, c))
            s = (a + d * k) % c
            t = (o + s).astype(np.float64)

            if component_weight_pareto_alpha > 0.0:
                w = float(1.0 + rng.pareto(component_weight_pareto_alpha))
            else:
                w = 1.0
            t_mix += w * t
            w_sum += w
            latents["c_vals"][i, j] = c
            latents["o_vals"][i, j] = o
            latents["d_vals"][i, j] = d
            latents["a_vals"][i, j] = a
            latents["weights"][i, j] = w

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

    return x.astype(np.int64), y.astype(np.int64), latents


def make_modarith_dataset_bundle(
    seq_len: int,
    vocab_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    zipf_c: float,
    zipf_o: float,
    c_min: int,
    min_step_frac: float,
    allow_noncoprime: bool,
    noncoprime_prob: float,
    mix_components_min: int,
    mix_components_max: int,
    component_weight_pareto_alpha: float,
    token_noise_std: float,
    token_noise_t_df: float,
    seed: int,
) -> ModArithDatasetBundle:
    rng = np.random.default_rng(seed + 303)

    x_train, y_train, latents_train = _sample_split(
        n=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        zipf_c=zipf_c,
        zipf_o=zipf_o,
        c_min=c_min,
        min_step_frac=min_step_frac,
        allow_noncoprime=allow_noncoprime,
        noncoprime_prob=noncoprime_prob,
        mix_components_min=mix_components_min,
        mix_components_max=mix_components_max,
        component_weight_pareto_alpha=component_weight_pareto_alpha,
        token_noise_std=token_noise_std,
        token_noise_t_df=token_noise_t_df,
        rng=rng,
    )
    x_val, y_val, latents_val = _sample_split(
        n=val_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        zipf_c=zipf_c,
        zipf_o=zipf_o,
        c_min=c_min,
        min_step_frac=min_step_frac,
        allow_noncoprime=allow_noncoprime,
        noncoprime_prob=noncoprime_prob,
        mix_components_min=mix_components_min,
        mix_components_max=mix_components_max,
        component_weight_pareto_alpha=component_weight_pareto_alpha,
        token_noise_std=token_noise_std,
        token_noise_t_df=token_noise_t_df,
        rng=rng,
    )
    x_test, y_test, latents_test = _sample_split(
        n=test_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        zipf_c=zipf_c,
        zipf_o=zipf_o,
        c_min=c_min,
        min_step_frac=min_step_frac,
        allow_noncoprime=allow_noncoprime,
        noncoprime_prob=noncoprime_prob,
        mix_components_min=mix_components_min,
        mix_components_max=mix_components_max,
        component_weight_pareto_alpha=component_weight_pareto_alpha,
        token_noise_std=token_noise_std,
        token_noise_t_df=token_noise_t_df,
        rng=rng,
    )

    metadata = {
        "task": "mod_arith_lm",
        "seed": int(seed),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "zipf_c": float(zipf_c),
        "zipf_o": float(zipf_o),
        "c_min": int(c_min),
        "min_step_frac": float(min_step_frac),
        "allow_noncoprime": bool(allow_noncoprime),
        "noncoprime_prob": float(noncoprime_prob),
        "mix_components_min": int(mix_components_min),
        "mix_components_max": int(mix_components_max),
        "component_weight_pareto_alpha": float(component_weight_pareto_alpha),
        "token_noise_std": float(token_noise_std),
        "token_noise_t_df": float(token_noise_t_df),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "latent_max_components": int(max(1, int(mix_components_max))),
        "latent_arrays_saved": True,
    }

    return ModArithDatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
        latents_train=latents_train,
        latents_val=latents_val,
        latents_test=latents_test,
    )


def export_modarith_dataset_bundle(bundle: ModArithDatasetBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {"x": bundle.x_train, "y": bundle.y_train}
    val_payload = {"x": bundle.x_val, "y": bundle.y_val}
    test_payload = {"x": bundle.x_test, "y": bundle.y_test}
    if bundle.latents_train:
        train_payload.update(bundle.latents_train)
    if bundle.latents_val:
        val_payload.update(bundle.latents_val)
    if bundle.latents_test:
        test_payload.update(bundle.latents_test)

    np.savez_compressed(out_dir / "train_split.npz", **train_payload)
    np.savez_compressed(out_dir / "val_split.npz", **val_payload)
    np.savez_compressed(out_dir / "test_split.npz", **test_payload)

    with (out_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2, sort_keys=True)


def load_modarith_dataset_bundle(path: Path) -> ModArithDatasetBundle:
    train = np.load(path / "train_split.npz")
    val = np.load(path / "val_split.npz")
    test = np.load(path / "test_split.npz")
    with (path / "dataset_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    def _read_latents(obj) -> Optional[Dict[str, np.ndarray]]:
        keys = {"component_count", "c_vals", "o_vals", "d_vals", "a_vals", "weights"}
        present = [k for k in keys if k in obj.files]
        if not present:
            return None
        return {k: obj[k] for k in present}

    return ModArithDatasetBundle(
        x_train=train["x"],
        y_train=train["y"],
        x_val=val["x"],
        y_val=val["y"],
        x_test=test["x"],
        y_test=test["y"],
        metadata=metadata,
        latents_train=_read_latents(train),
        latents_val=_read_latents(val),
        latents_test=_read_latents(test),
    )
