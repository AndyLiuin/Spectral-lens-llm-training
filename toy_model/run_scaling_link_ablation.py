from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import train_toy_run
    from .variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import train_toy_run
    from variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def scaling_curve(d: np.ndarray, a: float, s: float, c: float) -> np.ndarray:
    return a * np.power(d, -s) + c


def fit_scaling_exponent(d_values: np.ndarray, losses: np.ndarray) -> Tuple[float, float, float]:
    d = np.asarray(d_values, dtype=np.float64)
    l = np.asarray(losses, dtype=np.float64)
    order = np.argsort(d)
    d, l = d[order], l[order]

    c0 = max(0.0, float(np.min(l) * 0.9))
    a0 = max(1e-6, float(np.max(l) - c0))
    s0 = 0.5
    bounds = ([1e-8, 1e-4, 0.0], [1e6, 10.0, max(10.0, float(np.max(l) * 2.0))])
    try:
        popt, _ = curve_fit(scaling_curve, d, l, p0=[a0, s0, c0], bounds=bounds, maxfev=20000)
        a, s, c = [float(x) for x in popt]
    except Exception:
        # Fallback: assume c=0 and fit log-log slope.
        good = (d > 0) & (l > 0)
        if good.sum() < 2:
            return float("nan"), float("nan"), float("nan")
        slope, intercept = np.polyfit(np.log(d[good]), np.log(l[good]), deg=1)
        s = float(-slope)
        a = float(np.exp(intercept))
        c = 0.0
    return a, s, c


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    cors = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xb = x[idx]
        yb = y[idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        cors.append(float(np.corrcoef(xb, yb)[0, 1]))
    if not cors:
        return float("nan"), float("nan"), float("nan")
    return float(np.quantile(cors, 0.025)), float(np.quantile(cors, 0.5)), float(np.quantile(cors, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling-link ablation for toy RFF models.")
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a,b")
    parser.add_argument("--d-values", type=str, default="4,8,16")
    parser.add_argument("--beta-values", type=str, default="0.75,1.0,1.5,2.0")
    parser.add_argument("--P-values", type=str, default="256,512,1024")
    parser.add_argument("--D-values", type=str, default="2000,8000,32000,128000")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
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
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no cap")
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    tracks = [x.strip() for x in args.tracks.split(",") if x.strip()]
    d_values = parse_int_list(args.d_values)
    beta_values = parse_float_list(args.beta_values)
    p_values = parse_int_list(args.P_values)
    dset_values = parse_int_list(args.D_values)
    seeds = parse_int_list(args.seeds)
    settings = resolve_variant_settings(
        args.variant,
        optimizer_name=args.optimizer_name,
        num_layers=args.num_layers,
        window_size=args.window_size,
        attention_scale=args.attention_scale,
    )

    run_rows: List[dict] = []
    n_run = 0

    for track in tracks:
        for d in d_values:
            for beta in beta_values:
                for p in p_values:
                    for seed in seeds:
                        for dset in dset_values:
                            if args.max_runs > 0 and n_run >= args.max_runs:
                                break
                            cfg = RunConfig(
                                task=args.task,
                                track=track,
                                d=d,
                                beta=beta,
                                p=p,
                                D=dset,
                                seq_len=args.seq_len,
                                B=args.B,
                                lr=args.lr,
                                seed=seed,
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
                                measurement=MeasurementConfig(
                                    n_samples=1024,
                                    fixed_samples=True,
                                    trace_normalize=True,
                                ),
                                output_root=Path(args.out_root),
                            )
                            state = train_toy_run(
                                config=cfg,
                                ablation_name="scaling_link_ablation",
                                write_summary=True,
                                save_spectra=True,
                                device=args.device,
                            )
                            final = state.metrics_df.sort_values("step").iloc[-1]
                            run_rows.append(
                                {
                                    "track": track,
                                    "task": args.task,
                                    "variant": settings["variant"],
                                    "optimizer_name": settings["optimizer_name"],
                                    "d": d,
                                    "beta": beta,
                                    "P": p,
                                    "seed": seed,
                                    "D": dset,
                                    "B": args.B,
                                    "lr": args.lr,
                                    "final_loss": float(final["loss"]),
                                    "test_loss": float(state.test_loss),
                                    "alpha_head": float(final["alpha_head"]),
                                    "alpha_tail": float(final["alpha_tail"]),
                                    "rankme": float(final["rankme"]),
                                    "run_name": state.config.run_name(),
                                }
                            )
                            n_run += 1
                        if args.max_runs > 0 and n_run >= args.max_runs:
                            break
                    if args.max_runs > 0 and n_run >= args.max_runs:
                        break
                if args.max_runs > 0 and n_run >= args.max_runs:
                    break
            if args.max_runs > 0 and n_run >= args.max_runs:
                break
        if args.max_runs > 0 and n_run >= args.max_runs:
            break

    runs_df = pd.DataFrame(run_rows)
    fit_rows: List[dict] = []

    if not runs_df.empty:
        group_cols = ["task", "track", "variant", "optimizer_name", "d", "beta", "P", "seed"]
        for keys, group in runs_df.groupby(group_cols):
            if len(group) < 3:
                continue
            dvals = group["D"].to_numpy(dtype=float)
            losses = group["test_loss"].to_numpy(dtype=float)
            a, s, c = fit_scaling_exponent(dvals, losses)
            idx_max_d = group["D"].idxmax()
            alpha_proxy = float(runs_df.loc[idx_max_d, "alpha_tail"])
            fit_rows.append(
                {
                    "task": keys[0],
                    "track": keys[1],
                    "variant": keys[2],
                    "optimizer_name": keys[3],
                    "d": keys[4],
                    "beta": keys[5],
                    "P": keys[6],
                    "seed": keys[7],
                    "a": a,
                    "s": s,
                    "c": c,
                    "alpha_proxy": alpha_proxy,
                }
            )

    fit_df = pd.DataFrame(fit_rows)

    corr_summary = {
        "pearson_r": float("nan"),
        "ci_low": float("nan"),
        "ci_median": float("nan"),
        "ci_high": float("nan"),
        "n_groups": int(len(fit_df)),
    }

    if len(fit_df) >= 3:
        x = fit_df["s"].to_numpy(dtype=float)
        y = fit_df["alpha_proxy"].to_numpy(dtype=float)
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            corr_summary["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
            ci_lo, ci_med, ci_hi = bootstrap_corr(x, y, n_boot=1000, seed=0)
            corr_summary["ci_low"] = ci_lo
            corr_summary["ci_median"] = ci_med
            corr_summary["ci_high"] = ci_hi

    out_dir = Path(args.out_root) / "scaling_link_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_dir / "scaling_runs.csv", index=False)
    fit_df.to_csv(out_dir / "scaling_fit.csv", index=False)
    with (out_dir / "scaling_correlation.json").open("w", encoding="utf-8") as f:
        json.dump(corr_summary, f, indent=2, sort_keys=True)

    print(f"Wrote: {out_dir / 'scaling_runs.csv'}")
    print(f"Wrote: {out_dir / 'scaling_fit.csv'}")
    print(f"Wrote: {out_dir / 'scaling_correlation.json'}")


if __name__ == "__main__":
    main()
