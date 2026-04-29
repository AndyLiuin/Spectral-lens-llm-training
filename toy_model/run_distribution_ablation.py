from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import train_toy_run
    from .variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import train_toy_run
    from variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-distribution ablation for toy RFF models.")
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a,b")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--latent-dists", type=str, default="gaussian,uniform,student_t")
    parser.add_argument("--anisotropy-modes", type=str, default="isotropic,powerlaw")
    parser.add_argument("--anisotropy-gammas", type=str, default="0.8,1.2")
    parser.add_argument("--latent-df", type=float, default=3.0)
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

    tracks = parse_list(args.tracks)
    seeds = parse_int_list(args.seeds)
    latent_dists = parse_list(args.latent_dists)
    anisotropy_modes = parse_list(args.anisotropy_modes)
    gammas = parse_float_list(args.anisotropy_gammas)
    settings = resolve_variant_settings(
        args.variant,
        optimizer_name=args.optimizer_name,
        num_layers=args.num_layers,
        window_size=args.window_size,
        attention_scale=args.attention_scale,
    )

    final_rows = []

    for track in tracks:
        for seed in seeds:
            for dist in latent_dists:
                for aniso in anisotropy_modes:
                    gamma_vals = [1.0] if aniso == "isotropic" else gammas
                    for gamma in gamma_vals:
                        cfg = RunConfig(
                            task=args.task,
                            track=track,
                            d=args.d,
                            beta=args.beta,
                            p=args.P,
                            D=args.D,
                            seq_len=args.seq_len,
                            B=args.B,
                            lr=args.lr,
                            seed=seed,
                            latent_dist=dist,
                            latent_df=args.latent_df,
                            latent_anisotropy=aniso,
                            latent_anisotropy_gamma=gamma,
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
                            max_steps=args.max_steps,
                            num_layers=settings["num_layers"],
                            d_model=args.d_model,
                            n_heads=args.n_heads,
                            **optimizer_group_kwargs_from_args(args),
                            measurement=MeasurementConfig(n_samples=1024, fixed_samples=True, trace_normalize=True),
                            output_root=Path(args.out_root),
                        )
                        state = train_toy_run(
                            config=cfg,
                            ablation_name="distribution_ablation",
                            write_summary=True,
                            save_spectra=True,
                            device=args.device,
                        )
                        final = state.metrics_df.sort_values("step").iloc[-1]
                        final_rows.append(
                            {
                                "track": track,
                                "task": args.task,
                                "seed": seed,
                                "latent_dist": dist,
                                "latent_df": args.latent_df,
                                "latent_anisotropy": aniso,
                                "latent_anisotropy_gamma": gamma,
                                "variant": settings["variant"],
                                "optimizer_name": settings["optimizer_name"],
                                "d": args.d,
                                "beta": args.beta,
                                "P": args.P,
                                "D": args.D,
                                "B": args.B,
                                "lr": args.lr,
                                "step": int(final["step"]),
                                "loss": float(final["loss"]),
                                "rankme": float(final["rankme"]),
                                "alpha_head": float(final["alpha_head"]),
                                "alpha_tail": float(final["alpha_tail"]),
                                "grad_rankme": float(final["grad_rankme"]),
                                "grad_alpha_head": float(final["grad_alpha_head"]),
                                "grad_alpha_tail": float(final["grad_alpha_tail"]),
                                "top10": float(final["top10"]),
                                "n_samples_measurement": int(final["n_samples_measurement"]),
                                "test_loss": float(state.test_loss),
                                "run_name": state.config.run_name(),
                            }
                        )

    out_dir = Path(args.out_root) / "distribution_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(out_dir / "distribution_ablation_summary.csv", index=False)

    # Aggregate means for easier plot/paper table use.
    agg = (
        final_df.groupby(
            [
                "task",
                "track",
                "variant",
                "optimizer_name",
                "latent_dist",
                "latent_anisotropy",
                "latent_anisotropy_gamma",
            ],
            as_index=False,
        )
        .agg(
            loss_mean=("loss", "mean"),
            rankme_mean=("rankme", "mean"),
            alpha_tail_mean=("alpha_tail", "mean"),
            grad_alpha_tail_mean=("grad_alpha_tail", "mean"),
            n=("loss", "size"),
        )
        .sort_values(
            [
                "task",
                "track",
                "variant",
                "optimizer_name",
                "latent_dist",
                "latent_anisotropy",
                "latent_anisotropy_gamma",
            ]
        )
    )
    agg.to_csv(out_dir / "distribution_ablation_aggregate.csv", index=False)

    print(f"Wrote: {out_dir / 'distribution_ablation_summary.csv'}")
    print(f"Wrote: {out_dir / 'distribution_ablation_aggregate.csv'}")


if __name__ == "__main__":
    main()
