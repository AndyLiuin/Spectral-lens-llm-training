from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import matched_loss_rows, train_toy_run
    from .variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import matched_loss_rows, train_toy_run
    from variant_utils import add_optimizer_group_args, optimizer_group_kwargs_from_args, resolve_variant_settings


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    p = np.maximum(a[:n], 0.0)
    q = np.maximum(b[:n], 0.0)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)))
    kl_qm = np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))
    return float(0.5 * (kl_pm + kl_qm))


def common_loss_targets(dfs: List[pd.DataFrame], num_targets: int = 5) -> np.ndarray:
    mins, maxs = [], []
    pooled = []
    for df in dfs:
        x = df["loss"].dropna()
        if x.empty:
            continue
        arr = x.to_numpy(dtype=float)
        mins.append(float(np.min(arr)))
        maxs.append(float(np.max(arr)))
        pooled.append(arr)
    if not mins:
        return np.array([])
    lo = max(mins)
    hi = min(maxs)
    if hi <= lo:
        return np.array([])
    pooled_arr = np.concatenate(pooled, axis=0)
    pooled_arr = pooled_arr[(pooled_arr >= lo) & (pooled_arr <= hi)]
    if pooled_arr.size == 0:
        return np.array([])
    qs = np.linspace(0.0, 1.0, num=max(int(num_targets), 1))
    targets = np.quantile(pooled_arr, qs)
    return np.unique(targets.astype(float))


def load_cov_spectrum(run_dir: Path, step: int) -> np.ndarray:
    path = run_dir / "spectra" / f"cov_spectrum_step_{int(step):06d}.npy"
    if not path.exists():
        return np.array([])
    return np.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise-scale ablation for toy RFF models.")
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a,b")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--B-list", type=str, default="32,128,512")
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-target-loss", type=int, default=5)
    parser.add_argument(
        "--max-match-error",
        type=float,
        default=1e-3,
        help="Maximum allowed |matched_loss - target_loss| per run when computing matched-loss JS.",
    )
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
    b_values = parse_int_list(args.B_list)
    base_b = b_values[0]
    settings = resolve_variant_settings(
        args.variant,
        optimizer_name=args.optimizer_name,
        num_layers=args.num_layers,
        window_size=args.window_size,
        attention_scale=args.attention_scale,
    )

    run_registry: Dict[str, List[dict]] = {}

    for track in tracks:
        for regime in ("unconstrained", "fixed_noise"):
            key = f"{track}:{regime}"
            run_registry[key] = []
            for bsz in b_values:
                if regime == "unconstrained":
                    lr = args.base_lr
                else:
                    lr = args.base_lr * (bsz / base_b)

                cfg = RunConfig(
                    task=args.task,
                    track=track,
                    d=args.d,
                    beta=args.beta,
                    p=args.P,
                    D=args.D,
                    seq_len=args.seq_len,
                    B=bsz,
                    lr=lr,
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
                state = train_toy_run(
                    config=cfg,
                    ablation_name=f"noise_scale_{regime}",
                    write_summary=True,
                    save_spectra=True,
                    device=args.device,
                    return_model_and_data=False,
                )
                run_registry[key].append(
                    {
                        "track": track,
                        "task": args.task,
                        "variant": settings["variant"],
                        "optimizer_name": settings["optimizer_name"],
                        "regime": regime,
                        "B": bsz,
                        "lr": lr,
                        "run_dir": state.run_dir,
                        "df": state.metrics_df,
                    }
                )

    matched_rows: List[dict] = []
    compare_rows: List[dict] = []

    for track in tracks:
        regime_to_table = {}
        for regime in ("unconstrained", "fixed_noise"):
            key = f"{track}:{regime}"
            runs = run_registry[key]
            dfs = [x["df"] for x in runs]
            targets = common_loss_targets(dfs, num_targets=args.num_target_loss)
            if len(targets) == 0:
                continue

            per_run_matched = []
            for r in runs:
                matched = matched_loss_rows(r["df"], targets)
                matched["B"] = r["B"]
                matched["lr"] = r["lr"]
                matched["run_dir"] = str(r["run_dir"])
                per_run_matched.append(matched)

            table = pd.concat(per_run_matched, ignore_index=True)
            # Keep only targets that are truly matched across all runs.
            # Otherwise JS can be dominated by step-mismatch rather than spectrum/noise effects.
            err_by_target = table.groupby("matched_loss_target", as_index=False)["matched_loss_error"].max()
            valid_targets = err_by_target[err_by_target["matched_loss_error"] <= args.max_match_error][
                "matched_loss_target"
            ].to_numpy(dtype=float)
            if len(valid_targets) == 0:
                max_err = float(err_by_target["matched_loss_error"].max()) if not err_by_target.empty else float("nan")
                print(
                    f"Skipping {track}/{regime}: no valid matched-loss targets within "
                    f"--max-match-error={args.max_match_error:g} (best max error={max_err:.6g})."
                )
                continue
            table = table[table["matched_loss_target"].isin(valid_targets)].copy()
            targets = np.sort(valid_targets)
            regime_to_table[regime] = table

            for target in targets:
                subset = table[np.isclose(table["matched_loss_target"], target)]
                divergences = []
                entries = list(subset.to_dict(orient="records"))
                match_errors = [float(e["matched_loss_error"]) for e in entries if "matched_loss_error" in e]
                for left, right in combinations(entries, 2):
                    s1 = load_cov_spectrum(Path(left["run_dir"]), int(left["step"]))
                    s2 = load_cov_spectrum(Path(right["run_dir"]), int(right["step"]))
                    divergences.append(js_divergence(s1, s2))
                row = {
                    "track": track,
                    "task": args.task,
                    "variant": args.variant,
                    "optimizer_name": args.optimizer_name if args.variant != "muon" else "muon",
                    "regime": regime,
                    "matched_loss_target": float(target),
                    "mean_js_div": float(np.nanmean(divergences)) if divergences else float("nan"),
                    "std_js_div": float(np.nanstd(divergences)) if divergences else float("nan"),
                    "num_pairs": len(divergences),
                    "mean_match_error": float(np.nanmean(match_errors)) if match_errors else float("nan"),
                    "max_match_error": float(np.nanmax(match_errors)) if match_errors else float("nan"),
                }
                matched_rows.append(row)

        # Regime-level comparison summary.
        df_track = pd.DataFrame([r for r in matched_rows if r["track"] == track])
        if df_track.empty:
            continue
        unc = df_track[df_track["regime"] == "unconstrained"]["mean_js_div"].dropna().to_numpy()
        fix = df_track[df_track["regime"] == "fixed_noise"]["mean_js_div"].dropna().to_numpy()
        compare_rows.append(
            {
                "track": track,
                "variant": args.variant,
                "optimizer_name": args.optimizer_name if args.variant != "muon" else "muon",
                "unconstrained_mean_js": float(np.mean(unc)) if len(unc) else float("nan"),
                "fixed_noise_mean_js": float(np.mean(fix)) if len(fix) else float("nan"),
                "delta_js_fixed_minus_unconstrained": float(np.mean(fix) - np.mean(unc))
                if len(unc) and len(fix)
                else float("nan"),
            }
        )

    out_dir = Path(args.out_root) / "noise_scale_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_cols = [
        "task",
        "track",
        "regime",
        "matched_loss_target",
        "mean_js_div",
        "std_js_div",
        "num_pairs",
        "mean_match_error",
        "max_match_error",
    ]
    compare_cols = [
        "task",
        "track",
        "unconstrained_mean_js",
        "fixed_noise_mean_js",
        "delta_js_fixed_minus_unconstrained",
    ]
    pd.DataFrame(matched_rows, columns=matched_cols).to_csv(out_dir / "noise_scale_matched_loss.csv", index=False)
    pd.DataFrame(compare_rows, columns=compare_cols).to_csv(out_dir / "noise_scale_comparison.csv", index=False)

    print(f"Wrote: {out_dir / 'noise_scale_matched_loss.csv'}")
    print(f"Wrote: {out_dir / 'noise_scale_comparison.csv'}")


if __name__ == "__main__":
    main()
