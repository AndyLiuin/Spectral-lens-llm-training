from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import matched_loss_rows, train_toy_run
    from .variant_utils import (
        add_optimizer_group_args,
        cumulative_variant_combos,
        optimizer_group_kwargs_from_args,
        resolve_variant_settings,
    )
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import matched_loss_rows, train_toy_run
    from variant_utils import (
        add_optimizer_group_args,
        cumulative_variant_combos,
        optimizer_group_kwargs_from_args,
        resolve_variant_settings,
    )


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def load_cov_spectrum(run_dir: Path, step: int) -> np.ndarray:
    path = run_dir / "spectra" / f"cov_spectrum_step_{int(step):06d}.npy"
    if not path.exists():
        return np.array([])
    return np.load(path)


def common_loss_targets(dfs: List[pd.DataFrame], num_targets: int = 5) -> np.ndarray:
    mins, maxs = [], []
    for df in dfs:
        x = df["loss"].dropna()
        if x.empty:
            continue
        mins.append(float(x.min()))
        maxs.append(float(x.max()))
    if not mins:
        return np.array([])
    lo = max(mins)
    hi = min(maxs)
    if hi <= lo:
        return np.array([])
    return np.linspace(lo, hi, num=num_targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative variant ablation for toy models.")
    parser.add_argument("--task", type=str, default="mod_arith_lm", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a")
    parser.add_argument("--variant-order", type=str, default="baseline,rope,muon,unet,fixed_window,attn_scale")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--target-loss-metric", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--target-loss-patience", type=int, default=1)
    parser.add_argument("--target-loss-min-steps", type=int, default=0)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--save-param-spectra", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-svd-samples", type=int, default=32)
    parser.add_argument(
        "--param-spectrum-paths",
        type=str,
        default="",
        help="Optional comma-separated matrix paths (e.g., blocks.0.attn.c_proj.weight,blocks.0.mlp.c_proj.weight).",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lm-head-softcap", type=float, default=30.0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--attention-scale", type=float, default=0.12)
    add_optimizer_group_args(parser)
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
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--num-target-loss", type=int, default=5)
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    tracks = parse_list(args.tracks)
    seeds = parse_int_list(args.seeds)
    stages = cumulative_variant_combos(parse_list(args.variant_order))

    run_rows: List[dict] = []
    registry: Dict[tuple, dict] = {}

    for track in tracks:
        for variant_index, variant_stage, variant_combo in stages:
            settings = resolve_variant_settings(
                variant_combo,
                optimizer_name="adamw",
                num_layers=args.num_layers,
                window_size=args.window_size,
                attention_scale=args.attention_scale,
            )
            for seed in seeds:
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
                    **optimizer_group_kwargs_from_args(args),
                    max_steps=args.max_steps,
                    target_loss=args.target_loss,
                    target_loss_metric=args.target_loss_metric,
                    target_loss_patience=args.target_loss_patience,
                    target_loss_min_steps=args.target_loss_min_steps,
                    num_layers=settings["num_layers"],
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    lm_head_softcap=args.lm_head_softcap,
                    measurement=MeasurementConfig(n_samples=1024, fixed_samples=True, trace_normalize=True),
                    save_checkpoints=args.save_checkpoints,
                    checkpoint_every=args.checkpoint_every,
                    save_param_spectra=args.save_param_spectra,
                    grad_svd_samples=args.grad_svd_samples,
                    param_spectrum_paths=tuple(parse_list(args.param_spectrum_paths)),
                    output_root=Path(args.out_root),
                )
                state = train_toy_run(
                    config=cfg,
                    ablation_name="variant_concat_ablation",
                    write_summary=True,
                    save_spectra=True,
                    device=args.device,
                )
                final = state.metrics_df.sort_values("step").iloc[-1]
                row = {
                    "task": args.task,
                    "track": track,
                    "variant_stage": variant_stage,
                    "variant_index": variant_index,
                    "variant_combo": variant_combo,
                    "seed": seed,
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
                    "tokens_seen": float(final.get("tokens_seen", np.nan)),
                    "train_time_s": float(final.get("train_time_s", np.nan)),
                    "target_loss": float(args.target_loss) if args.target_loss is not None else np.nan,
                    "target_loss_metric": args.target_loss_metric,
                    "stop_reason": state.summary_row.get("stop_reason", "max_steps"),
                    "stopped_early": bool(state.summary_row.get("stopped_early", False)),
                    "final_step": int(state.summary_row.get("final_step", final["step"])),
                    "test_loss": float(state.test_loss),
                    "window_size": settings["window_size"],
                    "attention_scale": settings["attention_scale"],
                    "run_name": state.config.run_name(),
                    "run_dir": str(state.run_dir),
                }
                run_rows.append(row)
                registry[(track, variant_combo, seed)] = {
                    "metrics_df": state.metrics_df,
                    "run_dir": state.run_dir,
                    "variant_stage": variant_stage,
                    "variant_index": variant_index,
                }

    summary_df = pd.DataFrame(run_rows)

    matched_rows = []
    baseline_combo = stages[0][2] if stages else "baseline"
    for track in tracks:
        for _, variant_stage, variant_combo in stages[1:]:
            baseline_dfs = []
            variant_dfs = []
            for seed in seeds:
                kb = (track, baseline_combo, seed)
                kv = (track, variant_combo, seed)
                if kb in registry and kv in registry:
                    baseline_dfs.append(registry[kb]["metrics_df"])
                    variant_dfs.append(registry[kv]["metrics_df"])
            if not baseline_dfs or not variant_dfs:
                continue

            targets = common_loss_targets(baseline_dfs + variant_dfs, num_targets=args.num_target_loss)
            if len(targets) == 0:
                continue

            for seed in seeds:
                kb = (track, baseline_combo, seed)
                kv = (track, variant_combo, seed)
                if kb not in registry or kv not in registry:
                    continue
                db = matched_loss_rows(registry[kb]["metrics_df"], targets)
                dv = matched_loss_rows(registry[kv]["metrics_df"], targets)
                if db.empty or dv.empty:
                    continue
                for target in targets:
                    rb = db[np.isclose(db["matched_loss_target"], target)]
                    rv = dv[np.isclose(dv["matched_loss_target"], target)]
                    if rb.empty or rv.empty:
                        continue
                    sb = load_cov_spectrum(registry[kb]["run_dir"], int(rb.iloc[0]["step"]))
                    sv = load_cov_spectrum(registry[kv]["run_dir"], int(rv.iloc[0]["step"]))
                    matched_rows.append(
                        {
                            "track": track,
                            "variant_stage": variant_stage,
                            "variant_combo": variant_combo,
                            "seed": seed,
                            "matched_loss_target": float(target),
                            "js_div_baseline_vs_variant": js_divergence(sb, sv),
                            "baseline_step": int(rb.iloc[0]["step"]),
                            "variant_step": int(rv.iloc[0]["step"]),
                            "baseline_loss": float(rb.iloc[0]["loss"]),
                            "variant_loss": float(rv.iloc[0]["loss"]),
                            "baseline_tokens_seen": float(rb.iloc[0].get("tokens_seen", np.nan)),
                            "variant_tokens_seen": float(rv.iloc[0].get("tokens_seen", np.nan)),
                            "baseline_train_time_s": float(rb.iloc[0].get("train_time_s", np.nan)),
                            "variant_train_time_s": float(rv.iloc[0].get("train_time_s", np.nan)),
                        }
                    )

    matched_df = pd.DataFrame(matched_rows)
    agg_matched = (
        matched_df.groupby(["track", "variant_stage", "variant_combo", "matched_loss_target"], as_index=False)
        .agg(
            js_div_mean=("js_div_baseline_vs_variant", "mean"),
            js_div_std=("js_div_baseline_vs_variant", "std"),
            n=("js_div_baseline_vs_variant", "size"),
        )
        if not matched_df.empty
        else pd.DataFrame(
            columns=["track", "variant_stage", "variant_combo", "matched_loss_target", "js_div_mean", "js_div_std", "n"]
        )
    )

    out_dir = Path(args.out_root) / "variant_concat_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "variant_concat_ablation_summary.csv", index=False)
    matched_df.to_csv(out_dir / "variant_concat_ablation_matched_pairs.csv", index=False)
    agg_matched.to_csv(out_dir / "variant_concat_ablation_matched_aggregate.csv", index=False)

    print(f"Wrote: {out_dir / 'variant_concat_ablation_summary.csv'}")
    print(f"Wrote: {out_dir / 'variant_concat_ablation_matched_pairs.csv'}")
    print(f"Wrote: {out_dir / 'variant_concat_ablation_matched_aggregate.csv'}")


if __name__ == "__main__":
    main()
