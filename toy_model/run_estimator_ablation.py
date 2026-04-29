from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import measure_model, train_toy_run
    from .variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import measure_model, train_toy_run
    from variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def append_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimator ablation for toy RFF models.")
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a,b", help="Comma-separated tracks: a,b")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-list", type=str, default="128,256,512,1024,4096")
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

    n_list = parse_int_list(args.n_list)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    settings = resolve_variant_settings(
        args.variant,
        optimizer_name=args.optimizer_name,
        num_layers=args.num_layers,
        window_size=args.window_size,
        attention_scale=args.attention_scale,
    )

    all_rows: List[dict] = []

    for track in tracks:
        base_cfg = RunConfig(
            task=args.task,
            track=track,
            d=args.d,
            beta=args.beta,
            p=args.P,
            D=args.D,
            seq_len=args.seq_len,
            B=args.B,
            lr=args.lr,
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
            measurement=MeasurementConfig(n_samples=max(n_list), fixed_samples=True, trace_normalize=True),
            output_root=Path(args.out_root),
        )

        state = train_toy_run(
            config=base_cfg,
            ablation_name="estimator_ablation",
            write_summary=True,
            save_spectra=True,
            device=args.device,
            return_model_and_data=True,
        )
        assert state.model is not None and state.dataset is not None
        if args.task == "mod_arith_lm":
            x_meas = state.dataset.x_train
            y_meas = state.dataset.y_train
        else:
            x_meas = state.dataset.z_train
            y_meas = state.dataset.y_train

        for fixed_mode in (True, False):
            mode_name = "fixed" if fixed_mode else "nonfixed"
            for n in n_list:
                meas = replace(base_cfg.measurement, n_samples=n, fixed_samples=fixed_mode)
                measured = measure_model(
                    model=state.model,
                    x=x_meas,
                    y=y_meas,
                    measurement=meas,
                    step=base_cfg.max_steps,
                    seed=base_cfg.seed,
                    namespace=f"estimator_holdout:{state.config.run_name()}",
                    device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
                    task=args.task,
                    vocab_size=args.vocab_size,
                    modarith_measurement_pooling=args.modarith_measurement_pooling,
                )
                m = measured["metrics"]
                repr_dim = len(measured["act_spectrum"])
                n_effective = int(measured["n_effective_samples"])
                row = {
                    "ablation": "estimator",
                    "task": args.task,
                    "measurement_mode": mode_name,
                    "modarith_measurement_pooling": args.modarith_measurement_pooling,
                    "probe_regime": args.probe_regime,
                    "track": track,
                    "d": args.d,
                    "beta": args.beta,
                    "P": args.P,
                    "D": args.D,
                    "B": args.B,
                    "lr": args.lr,
                    "latent_dist": args.latent_dist,
                    "latent_df": args.latent_df,
                    "latent_anisotropy": args.latent_anisotropy,
                    "latent_anisotropy_gamma": args.latent_anisotropy_gamma,
                    "variant": settings["variant"],
                    "optimizer_name": settings["optimizer_name"],
                    "step": args.max_steps,
                    "loss": float(state.metrics_df["loss"].dropna().iloc[-1]),
                    "rankme": m["rankme"],
                    "alpha_head": m["alpha_head"],
                    "alpha_tail": m["alpha_tail"],
                    "grad_rankme": m["grad_rankme"],
                    "grad_alpha_head": m["grad_alpha_head"],
                    "grad_alpha_tail": m["grad_alpha_tail"],
                    "top10": m["top10"],
                    "n_samples_measurement": n_effective,
                    "d_over_n": float(repr_dim / max(n_effective, 1)),
                    "run_name": state.config.run_name(),
                }
                all_rows.append(row)

        ablation_df = pd.DataFrame([r for r in all_rows if r["track"] == track])
        ablation_df.to_csv(state.run_dir / "estimator_ablation.csv", index=False)

    out_csv = Path(args.out_root) / "estimator_ablation" / "estimator_ablation_summary.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    append_csv(Path(args.out_root) / "toy_summary.csv", all_rows)

    print(f"Wrote estimator ablation summary to {out_csv}")


if __name__ == "__main__":
    main()
