from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class DatasetBundle:
    z_train: np.ndarray
    y_train: np.ndarray
    z_val: np.ndarray
    y_val: np.ndarray
    z_test: np.ndarray
    y_test: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    teacher_a: np.ndarray
    metadata: Dict[str, object]


def _teacher_coefficients(p: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(1, p + 1, dtype=np.float64)
    magnitudes = ranks ** (-beta)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=p)
    coeff = signs * magnitudes
    coeff /= np.linalg.norm(coeff) + 1e-12
    return coeff.astype(np.float32)


def _anisotropy_scales(d: int, mode: str, gamma: float) -> np.ndarray:
    if mode == "isotropic":
        scales = np.ones(d, dtype=np.float64)
    elif mode == "powerlaw":
        ranks = np.arange(1, d + 1, dtype=np.float64)
        eig = ranks ** (-gamma)
        eig /= np.mean(eig)
        scales = np.sqrt(eig)
    else:
        raise ValueError(f"Unsupported latent anisotropy mode: {mode}")
    return scales.astype(np.float32)


def _sample_base_latent(
    n: int,
    seq_len: int,
    d: int,
    rng: np.random.Generator,
    latent_dist: str,
    latent_df: float,
) -> np.ndarray:
    if latent_dist == "gaussian":
        z = rng.normal(loc=0.0, scale=1.0, size=(n, seq_len, d))
    elif latent_dist == "uniform":
        # Uniform with Var=1 per coordinate.
        lim = np.sqrt(3.0)
        z = rng.uniform(low=-lim, high=lim, size=(n, seq_len, d))
    elif latent_dist == "student_t":
        if latent_df <= 2.0:
            raise ValueError("latent_df must be > 2 for finite variance student_t.")
        z = rng.standard_t(df=latent_df, size=(n, seq_len, d))
        z = z * np.sqrt((latent_df - 2.0) / latent_df)
    else:
        raise ValueError(f"Unsupported latent distribution: {latent_dist}")
    return z.astype(np.float32)


def build_teacher(d: int, p: int, beta: float, sigma: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(loc=0.0, scale=sigma, size=(p, d)).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(p,)).astype(np.float32)
    teacher_a = _teacher_coefficients(p=p, beta=beta, rng=rng)
    return {
        "omega": omega,
        "phase": phase,
        "teacher_a": teacher_a,
    }


def rff_features_numpy(z: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    # z: [..., d], omega: [P, d], phase: [P]
    z_flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
    feats = np.cos(z_flat @ omega.T + phase[None, :])
    feats *= np.sqrt(2.0 / omega.shape[0])
    return feats.reshape(*z.shape[:-1], omega.shape[0]).astype(np.float32)


def _make_split(
    n: int,
    seq_len: int,
    d: int,
    omega: np.ndarray,
    phase: np.ndarray,
    teacher_a: np.ndarray,
    noise_std: float,
    latent_dist: str,
    latent_df: float,
    latent_anisotropy: str,
    latent_anisotropy_gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    z = _sample_base_latent(
        n=n,
        seq_len=seq_len,
        d=d,
        rng=rng,
        latent_dist=latent_dist,
        latent_df=latent_df,
    )
    scales = _anisotropy_scales(d=d, mode=latent_anisotropy, gamma=latent_anisotropy_gamma)
    z = z * scales.reshape(1, 1, d)
    phi = rff_features_numpy(z, omega=omega, phase=phase)
    pooled = phi.mean(axis=1)
    y = pooled @ teacher_a
    if noise_std > 0:
        y = y + rng.normal(loc=0.0, scale=noise_std, size=(n,)).astype(np.float32)
    return z, y.astype(np.float32)


def make_dataset_bundle(
    d: int,
    p: int,
    beta: float,
    sigma: float,
    seq_len: int,
    train_size: int,
    val_size: int,
    test_size: int,
    noise_std: float,
    latent_dist: str,
    latent_df: float,
    latent_anisotropy: str,
    latent_anisotropy_gamma: float,
    seed: int,
) -> DatasetBundle:
    teacher = build_teacher(d=d, p=p, beta=beta, sigma=sigma, seed=seed)
    rng = np.random.default_rng(seed + 101)

    z_train, y_train = _make_split(
        n=train_size,
        seq_len=seq_len,
        d=d,
        omega=teacher["omega"],
        phase=teacher["phase"],
        teacher_a=teacher["teacher_a"],
        noise_std=noise_std,
        latent_dist=latent_dist,
        latent_df=latent_df,
        latent_anisotropy=latent_anisotropy,
        latent_anisotropy_gamma=latent_anisotropy_gamma,
        rng=rng,
    )
    z_val, y_val = _make_split(
        n=val_size,
        seq_len=seq_len,
        d=d,
        omega=teacher["omega"],
        phase=teacher["phase"],
        teacher_a=teacher["teacher_a"],
        noise_std=noise_std,
        latent_dist=latent_dist,
        latent_df=latent_df,
        latent_anisotropy=latent_anisotropy,
        latent_anisotropy_gamma=latent_anisotropy_gamma,
        rng=rng,
    )
    z_test, y_test = _make_split(
        n=test_size,
        seq_len=seq_len,
        d=d,
        omega=teacher["omega"],
        phase=teacher["phase"],
        teacher_a=teacher["teacher_a"],
        noise_std=noise_std,
        latent_dist=latent_dist,
        latent_df=latent_df,
        latent_anisotropy=latent_anisotropy,
        latent_anisotropy_gamma=latent_anisotropy_gamma,
        rng=rng,
    )

    metadata = {
        "d": int(d),
        "P": int(p),
        "beta": float(beta),
        "sigma": float(sigma),
        "seed": int(seed),
        "seq_len": int(seq_len),
        "noise_std": float(noise_std),
        "latent_dist": str(latent_dist),
        "latent_df": float(latent_df),
        "latent_anisotropy": str(latent_anisotropy),
        "latent_anisotropy_gamma": float(latent_anisotropy_gamma),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
    }

    return DatasetBundle(
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        z_test=z_test,
        y_test=y_test,
        omega=teacher["omega"],
        phase=teacher["phase"],
        teacher_a=teacher["teacher_a"],
        metadata=metadata,
    )


def export_dataset_bundle(bundle: DatasetBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "train_split.npz", z=bundle.z_train, y=bundle.y_train)
    np.savez_compressed(out_dir / "val_split.npz", z=bundle.z_val, y=bundle.y_val)
    np.savez_compressed(out_dir / "test_split.npz", z=bundle.z_test, y=bundle.y_test)
    np.savez_compressed(
        out_dir / "teacher_params.npz",
        omega=bundle.omega,
        phase=bundle.phase,
        teacher_a=bundle.teacher_a,
    )

    with (out_dir / "teacher_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2, sort_keys=True)
