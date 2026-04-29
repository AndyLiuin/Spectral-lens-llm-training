from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import train_toy_run
    from .variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import train_toy_run
    from variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings


def count_sign_changes(values: np.ndarray, tol: float = 1e-6) -> int:
    if values.size < 3:
        return 0
    d = np.diff(values)
    signs = np.sign(d)
    signs[np.abs(d) < tol] = 0
    # Drop zeros to avoid overcounting.
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def summarize_phase(df: pd.DataFrame, regime: str, task: str, track: str, variant: str, optimizer_name: str) -> dict:
    rank = df["rankme"].to_numpy(dtype=float)
    alpha = df["alpha_tail"].to_numpy(dtype=float)
    grad_top10 = df["grad_top10"].to_numpy(dtype=float) if "grad_top10" in df.columns else np.array([])
    return {
        "track": track,
        "task": task,
        "variant": variant,
        "optimizer_name": optimizer_name,
        "regime": regime,
        "rankme_min": float(np.nanmin(rank)) if rank.size else float("nan"),
        "rankme_max": float(np.nanmax(rank)) if rank.size else float("nan"),
        "rankme_delta": float(rank[-1] - rank[0]) if rank.size > 1 else float("nan"),
        "rankme_sign_changes": count_sign_changes(rank),
        "alpha_tail_delta": float(alpha[-1] - alpha[0]) if alpha.size > 1 else float("nan"),
        "grad_top10_delta": float(grad_top10[-1] - grad_top10[0]) if grad_top10.size > 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-like dynamics ablation for toy RFF models.")
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a,b")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--pre-batch", type=int, default=512)
    parser.add_argument("--pre-lr", type=float, default=2e-4)
    parser.add_argument("--post-batch", type=int, default=64)
    parser.add_argument("--post-lr", type=float, default=8e-4)
    parser.add_argument("--transition-frac", type=float, default=0.5)
    parser.add_argument("--latent-dist", type=str, default="gaussian", choices=["gaussian", "uniform", "student_t"])
    parser.add_argument("--latent-df", type=float, default=3.0)
    parser.add_argument("--latent-anisotropy", type=str, default="isotropic", choices=["isotropic", "powerlaw"])
    parser.add_argument("--latent-anisotropy-gamma", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--zipf-c", type=float, default=1.3)
    parser.add_argument("--zipf-o", type=float, default=1.2)
    parser.add_argument("--c-min", type=int, default=5)
    parser.add_argument("--min-step-frac", type=float, default=0.125)
    parser.add_argument("--allow-noncoprime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noncoprime-prob", type=float, default=0.3)
    parser.add_argument("--mix-components-min", type=int, default=1)
    parser.add_argument("--mix-components-max", type=int, default=1)
    parser.add_argument("--component-weight-pareto-alpha", type=float, default=0.0)
    parser.add_argument("--token-noise-std", type=float, default=0.0)
    parser.add_argument("--token-noise-t-df", type=float, default=0.0)
    parser.add_argument(
        "--modarith-measurement-pooling",
        type=str,
        default="last",
        choices=["token", "last", "mean"],
    )
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--optimizer-name", type=str, default="adamw", choices=["adamw", "muon"])
    add_optimizer_group_args(parser)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--attention-scale", type=float, default=0.12)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    tracks = [x.strip() for x in args.tracks.split(",") if x.strip()]
    transition_step = max(1, int(args.max_steps * args.transition_frac))
    settings = resolve_variant_settings(
        args.variant,
        optimizer_name=args.optimizer_name,
        num_layers=args.num_layers,
        window_size=args.window_size,
        attention_scale=args.attention_scale,
    )

    all_traj: List[pd.DataFrame] = []
    all_summary: List[dict] = []

    for track in tracks:
        phase_cfg = RunConfig(
            task=args.task,
            track=track,
            d=args.d,
            beta=args.beta,
            p=args.P,
            D=args.D,
            seq_len=args.seq_len,
            B=args.pre_batch,
            lr=args.pre_lr,
            seed=args.seed,
            max_steps=args.max_steps,
            latent_dist=args.latent_dist,
            latent_df=args.latent_df,
            latent_anisotropy=args.latent_anisotropy,
            latent_anisotropy_gamma=args.latent_anisotropy_gamma,
            vocab_size=args.vocab_size,
            zipf_c=args.zipf_c,
            zipf_o=args.zipf_o,
            c_min=args.c_min,
            min_step_frac=args.min_step_frac,
            allow_noncoprime=args.allow_noncoprime,
            noncoprime_prob=args.noncoprime_prob,
            mix_components_min=args.mix_components_min,
            mix_components_max=args.mix_components_max,
            component_weight_pareto_alpha=args.component_weight_pareto_alpha,
            token_noise_std=args.token_noise_std,
            token_noise_t_df=args.token_noise_t_df,
            modarith_measurement_pooling=args.modarith_measurement_pooling,
            probe_regime=args.probe_regime,
            variant=settings["variant"],
            optimizer_name=settings["optimizer_name"],
            window_size=settings["window_size"],
            attention_scale=settings["attention_scale"],
            num_layers=settings["num_layers"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            **optimizer_group_kwargs_from_args(args),
            transition_step=transition_step,
            transition_batch=args.post_batch,
            transition_lr=args.post_lr,
            measurement=MeasurementConfig(n_samples=1024, fixed_samples=True, trace_normalize=True),
            output_root=Path(args.out_root),
        )
        muted_cfg = RunConfig(
            task=args.task,
            track=track,
            d=args.d,
            beta=args.beta,
            p=args.P,
            D=args.D,
            seq_len=args.seq_len,
            B=args.pre_batch,
            lr=args.pre_lr,
            seed=args.seed,
            max_steps=args.max_steps,
            latent_dist=args.latent_dist,
            latent_df=args.latent_df,
            latent_anisotropy=args.latent_anisotropy,
            latent_anisotropy_gamma=args.latent_anisotropy_gamma,
            vocab_size=args.vocab_size,
            zipf_c=args.zipf_c,
            zipf_o=args.zipf_o,
            c_min=args.c_min,
            min_step_frac=args.min_step_frac,
            allow_noncoprime=args.allow_noncoprime,
            noncoprime_prob=args.noncoprime_prob,
            mix_components_min=args.mix_components_min,
            mix_components_max=args.mix_components_max,
            component_weight_pareto_alpha=args.component_weight_pareto_alpha,
            token_noise_std=args.token_noise_std,
            token_noise_t_df=args.token_noise_t_df,
            modarith_measurement_pooling=args.modarith_measurement_pooling,
            probe_regime=args.probe_regime,
            variant=settings["variant"],
            optimizer_name=settings["optimizer_name"],
            window_size=settings["window_size"],
            attention_scale=settings["attention_scale"],
            num_layers=settings["num_layers"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            **optimizer_group_kwargs_from_args(args),
            measurement=MeasurementConfig(n_samples=1024, fixed_samples=True, trace_normalize=True),
            output_root=Path(args.out_root),
        )

        phase_state = train_toy_run(
            config=phase_cfg,
            ablation_name="phase_ablation_phase_like",
            write_summary=True,
            save_spectra=True,
            device=args.device,
        )
        muted_state = train_toy_run(
            config=muted_cfg,
            ablation_name="phase_ablation_muted",
            write_summary=True,
            save_spectra=True,
            device=args.device,
        )

        for regime, state in (("phase_like", phase_state), ("muted", muted_state)):
            df = state.metrics_df.copy()
            df["regime"] = regime
            df["track"] = track
            all_traj.append(df)
            all_summary.append(
                summarize_phase(
                    df,
                    regime=regime,
                    task=args.task,
                    track=track,
                    variant=settings["variant"],
                    optimizer_name=settings["optimizer_name"],
                )
            )

    out_dir = Path(args.out_root) / "phase_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_traj, ignore_index=True).to_csv(out_dir / "phase_trajectories.csv", index=False)
    pd.DataFrame(all_summary).to_csv(out_dir / "phase_summary.csv", index=False)

    print(f"Wrote: {out_dir / 'phase_trajectories.csv'}")
    print(f"Wrote: {out_dir / 'phase_summary.csv'}")


if __name__ == "__main__":
    main()
