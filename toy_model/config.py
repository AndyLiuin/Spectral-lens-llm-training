from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AblationGrid:
    d_values: Tuple[int, ...] = (4, 8, 16)
    beta_values: Tuple[float, ...] = (0.75, 1.0, 1.5, 2.0)
    p_values: Tuple[int, ...] = (256, 512, 1024)
    dset_values: Tuple[int, ...] = (2000, 8000, 32000, 128000)
    seeds: Tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class MeasurementConfig:
    n_samples: int = 512
    fixed_samples: bool = True
    trace_normalize: bool = True
    alpha_head_range: Tuple[int, int] = (1, 10)
    alpha_tail_range: Tuple[int, int] = (50, 200)


@dataclass(frozen=True)
class RunConfig:
    track: str
    d: int
    beta: float
    p: int
    D: int
    B: int
    lr: float
    seed: int
    task: str = "rff_regression"
    sigma: float = 1.0
    noise_std: float = 0.02
    latent_dist: str = "gaussian"
    latent_df: float = 3.0
    latent_anisotropy: str = "isotropic"
    latent_anisotropy_gamma: float = 1.0
    variant: str = "baseline"
    optimizer_name: str = "adamw"
    window_size: int = 0
    attention_scale: Optional[float] = None
    embed_lr: Optional[float] = None
    head_lr: Optional[float] = None
    scalar_lr: Optional[float] = None
    muon_lr: Optional[float] = None
    muon_momentum: float = 0.95
    muon_backend: str = "newtonschulz5"
    muon_backend_steps: int = 5
    muon_notebook_lr_defaults: bool = True
    seq_len: int = 16
    vocab_size: int = 1024
    zipf_c: float = 1.3
    zipf_o: float = 1.2
    c_min: int = 5
    min_step_frac: float = 0.125
    allow_noncoprime: bool = True
    noncoprime_prob: float = 0.3
    mix_components_min: int = 1
    mix_components_max: int = 1
    component_weight_pareto_alpha: float = 0.0
    token_noise_std: float = 0.0
    token_noise_t_df: float = 0.0
    modarith_measurement_pooling: str = "last"
    probe_regime: str = "both"
    val_size: int = 2048
    test_size: int = 4096
    max_steps: int = 1200
    target_loss: Optional[float] = None
    target_loss_metric: str = "val"
    target_loss_patience: int = 1
    target_loss_min_steps: int = 0
    log_every: int = 50
    eval_every: int = 50
    measurement_every: int = 50
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    save_checkpoints: bool = False
    checkpoint_every: int = 0
    checkpoint_steps: Tuple[int, ...] = ()
    save_param_spectra: bool = False
    grad_svd_samples: int = 32
    param_spectrum_paths: Tuple[str, ...] = ()
    transition_step: Optional[int] = None
    transition_batch: Optional[int] = None
    transition_lr: Optional[float] = None
    num_layers: int = 1
    d_model: int = 128
    n_heads: int = 4
    ff_mult: int = 2
    lm_head_softcap: float = 30.0
    weight_decay: float = 0.0
    output_root: Path = Path("toy_outputs")

    def run_name(self) -> str:
        parts = [
            f"track-{self.track}",
            f"d-{self.d}",
            f"beta-{self.beta}",
            f"P-{self.p}",
            f"D-{self.D}",
            f"B-{self.B}",
            f"lr-{self.lr:g}",
            f"seed-{self.seed}",
            f"dist-{self.latent_dist}",
            f"aniso-{self.latent_anisotropy}",
            f"var-{self.variant}",
            f"steps-{self.max_steps}",
        ]
        if self.latent_dist == "student_t":
            parts.append(f"df-{self.latent_df:g}")
        if self.latent_anisotropy != "isotropic":
            parts.append(f"gamma-{self.latent_anisotropy_gamma:g}")
        if self.window_size > 0:
            parts.append(f"win-{self.window_size}")
        if self.attention_scale is not None:
            parts.append(f"as-{self.attention_scale:g}")
        if self.embed_lr is not None:
            parts.append(f"elr-{self.embed_lr:g}")
        if self.head_lr is not None:
            parts.append(f"hlr-{self.head_lr:g}")
        if self.scalar_lr is not None:
            parts.append(f"slr-{self.scalar_lr:g}")
        if self.muon_lr is not None:
            parts.append(f"mlr-{self.muon_lr:g}")
        if self.muon_momentum != 0.95:
            parts.append(f"mmom-{self.muon_momentum:g}")
        if self.muon_backend != "newtonschulz5":
            parts.append(f"mback-{self.muon_backend}")
        if self.muon_backend_steps != 5:
            parts.append(f"msteps-{self.muon_backend_steps}")
        if not self.muon_notebook_lr_defaults:
            parts.append("mnbdefaults-off")
        if self.lm_head_softcap != 30.0:
            parts.append(f"lmsc-{self.lm_head_softcap:g}")
        if self.weight_decay != 0.0:
            parts.append(f"wd-{self.weight_decay:g}")
        if self.task != "rff_regression":
            parts.append(f"task-{self.task}")
        if self.task == "mod_arith_lm":
            parts.extend(
                [
                    f"V-{self.vocab_size}",
                    f"zc-{self.zipf_c:g}",
                    f"zo-{self.zipf_o:g}",
                    f"cmin-{self.c_min}",
                    f"msf-{self.min_step_frac:g}",
                    f"ncp-{self.noncoprime_prob:g}" if self.allow_noncoprime else "ncp-0",
                    f"mpool-{self.modarith_measurement_pooling}",
                    f"probe-{self.probe_regime}",
                ]
            )
            if self.mix_components_max > 1 or self.mix_components_min > 1:
                parts.append(f"mix-{self.mix_components_min}-{self.mix_components_max}")
            if self.component_weight_pareto_alpha > 0:
                parts.append(f"palpha-{self.component_weight_pareto_alpha:g}")
            if self.token_noise_std > 0:
                parts.append(f"nstd-{self.token_noise_std:g}")
            if self.token_noise_t_df > 0:
                parts.append(f"ndf-{self.token_noise_t_df:g}")
        if self.transition_step is not None:
            parts.append(f"trstep-{self.transition_step}")
            parts.append(f"trB-{self.transition_batch}")
            parts.append(f"trlr-{self.transition_lr:g}")
        if self.target_loss is not None and np.isfinite(float(self.target_loss)):
            parts.append(f"tloss-{float(self.target_loss):g}")
            parts.append(f"tmetric-{self.target_loss_metric}")
            if self.target_loss_patience > 1:
                parts.append(f"tpat-{self.target_loss_patience}")
            if self.target_loss_min_steps > 0:
                parts.append(f"tmin-{self.target_loss_min_steps}")
        if self.save_checkpoints:
            if self.checkpoint_every > 0:
                parts.append(f"ckpt-{self.checkpoint_every}")
            if self.checkpoint_steps:
                ckpt_steps = tuple(int(x) for x in self.checkpoint_steps)
                digest = hashlib.sha1(",".join(str(x) for x in ckpt_steps).encode("utf-8")).hexdigest()[:8]
                parts.append(f"ckpts-{len(ckpt_steps)}")
                parts.append(f"ckplan-{digest}")
        if self.save_param_spectra:
            parts.append(f"pspec-{self.grad_svd_samples}")
        if self.optimizer_name == "muon" or "muon" in self.variant.lower():
            parts.append("oprof-muon-split")
        full_name = "__".join(parts)
        if len(full_name) <= 180:
            return full_name

        digest = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:12]
        compact_parts = [
            f"track-{self.track}",
            f"D-{self.D}",
            f"B-{self.B}",
            f"seed-{self.seed}",
            f"var-{self.variant}",
            f"steps-{self.max_steps}",
        ]
        if self.task != "rff_regression":
            compact_parts.append(f"task-{self.task}")
        if self.task == "mod_arith_lm":
            compact_parts.extend(
                [
                    f"V-{self.vocab_size}",
                    f"mpool-{self.modarith_measurement_pooling}",
                ]
            )
        compact_parts.append(f"h-{digest}")
        return "__".join(compact_parts)


DEFAULT_COLUMNS: Sequence[str] = (
    "task",
    "track",
    "d",
    "beta",
    "P",
    "D",
    "B",
    "lr",
    "step",
    "loss",
    "rankme",
    "alpha_head",
    "alpha_tail",
    "grad_rankme",
    "grad_alpha_head",
    "grad_alpha_tail",
    "top10",
    "n_samples_measurement",
)
