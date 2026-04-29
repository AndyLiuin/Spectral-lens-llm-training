from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_alpha, compute_rankme, normalize_trace


def parse_steps(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    return sorted(set(out))


def parse_list(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        s = part.strip()
        if s:
            out.append(s)
    return out


def _resolve_run_dir(row: pd.Series, ablation_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    run_name = str(row.get("run_name", "")).strip()
    if run_name:
        candidates.append(ablation_dir / run_name)
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir:
        p = Path(run_dir)
        candidates.append(p if p.is_absolute() else (ablation_dir.parent.parent / p))
    for c in candidates:
        if c.exists():
            return c
    return None


def _alpha_pair(spec: np.ndarray, head: Tuple[int, int], tail: Tuple[int, int]) -> Tuple[float, float]:
    s = np.asarray(spec, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("nan"), float("nan")
    s = np.maximum(s, 0.0)
    if float(np.sum(s)) <= 0.0:
        return float("nan"), float("nan")
    s = normalize_trace(s)
    return float(compute_alpha(s, int(head[0]), int(head[1]))), float(compute_alpha(s, int(tail[0]), int(tail[1])))


def _load_spectrum(run_dir: Path, step: int, kind: str) -> Optional[np.ndarray]:
    path = run_dir / "spectra" / f"{kind}_spectrum_step_{int(step):06d}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=np.float64)


def _safe_mean_spectra(specs: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(s, dtype=np.float64).reshape(-1) for s in specs if s is not None and len(s) > 0]
    if not valid:
        return np.array([], dtype=np.float64)
    min_len = min(len(s) for s in valid)
    if min_len <= 0:
        return np.array([], dtype=np.float64)
    stacked = np.stack([normalize_trace(np.maximum(s[:min_len], 0.0)) for s in valid], axis=0)
    return stacked.mean(axis=0)


def _aggregate_mean_std(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    agg = df.groupby(list(group_cols), as_index=False).agg({c: ["mean", "std"] for c in value_cols})
    agg.columns = ["_".join([x for x in col if x]).rstrip("_") for col in agg.columns.to_flat_index()]
    return agg


def _plot_mean_std_lines(
    agg_df: pd.DataFrame,
    x_col: str,
    stage_col: str,
    metrics: Sequence[Tuple[str, str]],
    variant_order: Sequence[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    colors = {
        "baseline": "#7f7f7f",
        "rope": "#1f77b4",
        "muon": "#2ca02c",
        "unet": "#d62728",
        "fixed_window": "#ff7f0e",
        "attn_scale": "#9467bd",
    }
    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.8 * nrows), squeeze=False)
    for idx, (metric, label) in enumerate(metrics):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        mcol = f"{metric}_mean"
        scol = f"{metric}_std"
        for stage in variant_order:
            sub = agg_df[agg_df[stage_col] == stage].sort_values(x_col)
            if sub.empty or mcol not in sub.columns:
                continue
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[mcol].to_numpy(dtype=float)
            s = sub[scol].fillna(0.0).to_numpy(dtype=float) if scol in sub.columns else np.zeros_like(y)
            ax.plot(x, y, marker="o", linewidth=1.7, color=colors.get(stage, None), label=stage)
            ax.fill_between(x, y - s, y + s, color=colors.get(stage, None), alpha=0.16)
        ax.set_title(f"{title_prefix}: {label}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2)
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stage_spectrum_evolution(
    stage: str,
    steps: Sequence[int],
    stage_seed_run: Dict[Tuple[str, int], Path],
    seeds: Sequence[int],
    out_path: Path,
) -> None:
    cmap = plt.cm.plasma(np.linspace(0, 1, len(steps)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for i, step in enumerate(steps):
        act_specs: List[np.ndarray] = []
        grad_specs: List[np.ndarray] = []
        for seed in seeds:
            run_dir = stage_seed_run.get((stage, int(seed)))
            if run_dir is None:
                continue
            s_act = _load_spectrum(run_dir=run_dir, step=int(step), kind="cov")
            s_grad = _load_spectrum(run_dir=run_dir, step=int(step), kind="grad")
            if s_act is not None:
                act_specs.append(s_act)
            if s_grad is not None:
                grad_specs.append(s_grad)
        act_mean = _safe_mean_spectra(act_specs)
        grad_mean = _safe_mean_spectra(grad_specs)
        if act_mean.size > 0:
            axes[0].loglog(np.arange(1, len(act_mean) + 1), np.maximum(act_mean, 1e-20), color=cmap[i], linewidth=2.0, label=f"step {step}")
        if grad_mean.size > 0:
            axes[1].loglog(np.arange(1, len(grad_mean) + 1), np.maximum(grad_mean, 1e-20), color=cmap[i], linewidth=2.0, label=f"step {step}")

    axes[0].set_title(f"{stage}: Activation Covariance Spectrum")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Trace-normalized eigenvalue")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{stage}: Gradient Covariance Spectrum")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Trace-normalized eigenvalue")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _matched_loss_rows(df: pd.DataFrame, target_losses: Iterable[float]) -> pd.DataFrame:
    rows: List[dict] = []
    if df.empty:
        return pd.DataFrame(rows)
    sdf = df.sort_values("step").copy()
    for target in target_losses:
        idx = (sdf["loss"] - float(target)).abs().idxmin()
        row = sdf.loc[idx].copy()
        row["matched_loss_target"] = float(target)
        row["matched_loss_error"] = float(abs(float(row["loss"]) - float(target)))
        rows.append(dict(row))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Variant-wide spectrum/stat evolution diagnostics for toy modarith concat runs.")
    parser.add_argument("--ablation-dir", type=str, default="toy_model/variant_concat_ablation_new")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--spectrum-steps", type=str, default="0,50,200,400")
    parser.add_argument(
        "--matrices",
        type=str,
        default="blocks_0_attn_c_proj_weight,blocks_0_mlp_c_proj_weight,blocks_1_attn_c_proj_weight,blocks_1_mlp_c_proj_weight",
    )
    parser.add_argument("--alpha-head", type=str, default="1,10")
    parser.add_argument("--alpha-tail", type=str, default="50,200")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ablation_dir / "analysis_deep")
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_head = tuple(int(x.strip()) for x in str(args.alpha_head).split(","))
    alpha_tail = tuple(int(x.strip()) for x in str(args.alpha_tail).split(","))
    requested_spec_steps = parse_steps(args.spectrum_steps)
    matrix_names = parse_list(args.matrices)

    summary_path = ablation_dir / "variant_concat_ablation_summary.csv"
    summary_df = pd.read_csv(summary_path).sort_values(["variant_index", "seed"])
    matched_pairs_path = ablation_dir / "variant_concat_ablation_matched_pairs.csv"
    matched_pairs_df = pd.read_csv(matched_pairs_path) if matched_pairs_path.exists() else pd.DataFrame()

    variant_order = [str(v) for v in summary_df.sort_values("variant_index")["variant_stage"].drop_duplicates().tolist()]
    seeds = sorted(int(s) for s in summary_df["seed"].drop_duplicates().tolist())

    stage_seed_run: Dict[Tuple[str, int], Path] = {}
    metrics_rows: List[pd.DataFrame] = []
    param_rows: List[pd.DataFrame] = []
    spectrum_rows: List[dict] = []

    for _, row in summary_df.iterrows():
        stage = str(row["variant_stage"])
        seed = int(row["seed"])
        run_dir = _resolve_run_dir(row=row, ablation_dir=ablation_dir)
        if run_dir is None:
            continue
        stage_seed_run[(stage, seed)] = run_dir

        metrics_path = run_dir / "metrics_over_time.csv"
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path).sort_values("step")
            mdf["variant_stage"] = stage
            mdf["variant_index"] = int(row["variant_index"])
            mdf["seed"] = seed
            mdf["run_name"] = str(row["run_name"])
            metrics_rows.append(mdf)

        param_path = run_dir / "param_spectra_over_time.csv"
        if param_path.exists():
            pdf = pd.read_csv(param_path).sort_values(["step", "spectrum_kind", "matrix_name"])
            pdf["variant_stage"] = stage
            pdf["variant_index"] = int(row["variant_index"])
            pdf["seed"] = seed
            pdf["run_name"] = str(row["run_name"])
            param_rows.append(pdf)

    if not metrics_rows:
        raise RuntimeError("No metrics_over_time.csv found under ablation dir.")

    metrics_all = pd.concat(metrics_rows, ignore_index=True)
    metrics_all.to_csv(out_dir / "method_metrics_all_runs.csv", index=False)

    metric_cols = ["rankme", "alpha_head", "alpha_tail", "top10", "grad_rankme", "grad_alpha_head", "grad_alpha_tail", "grad_top10", "loss"]
    step_agg = _aggregate_mean_std(
        metrics_all,
        group_cols=["variant_stage", "step"],
        value_cols=metric_cols,
    )
    step_agg.to_csv(out_dir / "method_metrics_step_aggregate.csv", index=False)
    _plot_mean_std_lines(
        agg_df=step_agg,
        x_col="step",
        stage_col="variant_stage",
        metrics=[
            ("loss", "Val Loss"),
            ("rankme", "Activation RankMe"),
            ("alpha_head", "Activation Alpha Head"),
            ("alpha_tail", "Activation Alpha Tail"),
            ("grad_rankme", "Gradient RankMe"),
            ("grad_alpha_head", "Gradient Alpha Head"),
            ("grad_alpha_tail", "Gradient Alpha Tail"),
            ("grad_top10", "Gradient Top10 Share"),
        ],
        variant_order=variant_order,
        out_path=out_dir / "method_metrics_step_evolution.png",
        title_prefix="Step-Aligned Evolution",
    )

    # Loss-aligned trajectories (match main-paper style fairness).
    if matched_pairs_df is not None and not matched_pairs_df.empty and "matched_loss_target" in matched_pairs_df.columns:
        targets = sorted(float(x) for x in matched_pairs_df["matched_loss_target"].dropna().unique().tolist())
    else:
        losses = np.sort(metrics_all["loss"].dropna().to_numpy(dtype=float))
        targets = [float(x) for x in np.quantile(losses, [0.2, 0.4, 0.6, 0.8])]

    matched_rows: List[pd.DataFrame] = []
    for (stage, seed), g in metrics_all.groupby(["variant_stage", "seed"]):
        m = _matched_loss_rows(g, targets)
        if m.empty:
            continue
        m["variant_stage"] = str(stage)
        m["seed"] = int(seed)
        matched_rows.append(m)
    matched_all = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()
    if not matched_all.empty:
        matched_all.to_csv(out_dir / "method_metrics_loss_aligned_all.csv", index=False)
        loss_agg = _aggregate_mean_std(
            matched_all,
            group_cols=["variant_stage", "matched_loss_target"],
            value_cols=["rankme", "alpha_head", "alpha_tail", "grad_rankme", "grad_alpha_head", "grad_alpha_tail", "grad_top10", "matched_loss_error"],
        )
        loss_agg.to_csv(out_dir / "method_metrics_loss_aligned_aggregate.csv", index=False)
        _plot_mean_std_lines(
            agg_df=loss_agg,
            x_col="matched_loss_target",
            stage_col="variant_stage",
            metrics=[
                ("rankme", "Activation RankMe"),
                ("alpha_head", "Activation Alpha Head"),
                ("alpha_tail", "Activation Alpha Tail"),
                ("grad_rankme", "Gradient RankMe"),
                ("grad_alpha_head", "Gradient Alpha Head"),
                ("grad_alpha_tail", "Gradient Alpha Tail"),
                ("grad_top10", "Gradient Top10 Share"),
            ],
            variant_order=variant_order,
            out_path=out_dir / "method_metrics_loss_aligned_evolution.png",
            title_prefix="Loss-Aligned Evolution",
        )

    # Per-stage spectrum-shape evolution for activation covariance and gradient covariance.
    for stage in variant_order:
        common_steps: Optional[set[int]] = None
        for seed in seeds:
            run_dir = stage_seed_run.get((stage, seed))
            if run_dir is None:
                continue
            mpath = run_dir / "metrics_over_time.csv"
            if not mpath.exists():
                continue
            steps = set(pd.read_csv(mpath)["step"].astype(int).tolist())
            common_steps = steps if common_steps is None else (common_steps & steps)
        if not common_steps:
            continue
        use_steps = [s for s in requested_spec_steps if s in common_steps]
        final_common = int(max(common_steps))
        if final_common not in use_steps:
            use_steps.append(final_common)
        use_steps = sorted(set(use_steps))

        _plot_stage_spectrum_evolution(
            stage=stage,
            steps=use_steps,
            stage_seed_run=stage_seed_run,
            seeds=seeds,
            out_path=out_dir / f"spectrum_evolution_{stage}.png",
        )

        for step in use_steps:
            act_specs: List[np.ndarray] = []
            grad_specs: List[np.ndarray] = []
            for seed in seeds:
                run_dir = stage_seed_run.get((stage, seed))
                if run_dir is None:
                    continue
                a = _load_spectrum(run_dir=run_dir, step=step, kind="cov")
                g = _load_spectrum(run_dir=run_dir, step=step, kind="grad")
                if a is not None:
                    act_specs.append(a)
                if g is not None:
                    grad_specs.append(g)
            act_m = _safe_mean_spectra(act_specs)
            grad_m = _safe_mean_spectra(grad_specs)
            if act_m.size > 0:
                ah, at = _alpha_pair(act_m, alpha_head, alpha_tail)
                spectrum_rows.append(
                    {
                        "variant_stage": stage,
                        "step": int(step),
                        "spectrum_kind": "activation_covariance",
                        "rankme": float(compute_rankme(act_m)),
                        "alpha_head": ah,
                        "alpha_tail": at,
                    }
                )
            if grad_m.size > 0:
                gh, gt = _alpha_pair(grad_m, alpha_head, alpha_tail)
                spectrum_rows.append(
                    {
                        "variant_stage": stage,
                        "step": int(step),
                        "spectrum_kind": "gradient_covariance",
                        "rankme": float(compute_rankme(grad_m)),
                        "alpha_head": gh,
                        "alpha_tail": gt,
                    }
                )

    pd.DataFrame(spectrum_rows).to_csv(out_dir / "spectrum_evolution_snapshot_stats.csv", index=False)

    # Matrix-level SVD stats evolution.
    if param_rows:
        params_all = pd.concat(param_rows, ignore_index=True)
        params_all.to_csv(out_dir / "matrix_svd_stats_all_runs.csv", index=False)
        params_sel = params_all[params_all["matrix_name"].isin(matrix_names)].copy()
        params_agg = _aggregate_mean_std(
            params_sel,
            group_cols=["variant_stage", "step", "spectrum_kind", "matrix_name"],
            value_cols=["spectrum_rankme", "spectrum_alpha_head", "spectrum_alpha_tail", "spectrum_top10"],
        )
        params_agg.to_csv(out_dir / "matrix_svd_stats_step_aggregate.csv", index=False)

        for matrix in matrix_names:
            for kind in ["weight_svd", "grad_svd"]:
                sub = params_agg[(params_agg["matrix_name"] == matrix) & (params_agg["spectrum_kind"] == kind)].copy()
                if sub.empty:
                    continue
                _plot_mean_std_lines(
                    agg_df=sub.rename(
                        columns={
                            "spectrum_rankme_mean": "rankme_mean",
                            "spectrum_rankme_std": "rankme_std",
                            "spectrum_alpha_head_mean": "alpha_head_mean",
                            "spectrum_alpha_head_std": "alpha_head_std",
                            "spectrum_alpha_tail_mean": "alpha_tail_mean",
                            "spectrum_alpha_tail_std": "alpha_tail_std",
                            "spectrum_top10_mean": "top10_mean",
                            "spectrum_top10_std": "top10_std",
                        }
                    ),
                    x_col="step",
                    stage_col="variant_stage",
                    metrics=[
                        ("rankme", "RankMe"),
                        ("alpha_head", "Alpha Head"),
                        ("alpha_tail", "Alpha Tail"),
                        ("top10", "Top10 Share"),
                    ],
                    variant_order=variant_order,
                    out_path=out_dir / f"matrix_{matrix}_{kind}_stats_evolution.png",
                    title_prefix=f"{kind}:{matrix}",
                )

    print(f"Wrote: {out_dir / 'method_metrics_all_runs.csv'}")
    print(f"Wrote: {out_dir / 'method_metrics_step_aggregate.csv'}")
    print(f"Wrote: {out_dir / 'method_metrics_step_evolution.png'}")
    if (out_dir / "method_metrics_loss_aligned_aggregate.csv").exists():
        print(f"Wrote: {out_dir / 'method_metrics_loss_aligned_aggregate.csv'}")
        print(f"Wrote: {out_dir / 'method_metrics_loss_aligned_evolution.png'}")
    print(f"Wrote: {out_dir / 'spectrum_evolution_snapshot_stats.csv'}")
    print(f"Wrote per-stage spectra: spectrum_evolution_<variant>.png")
    if (out_dir / "matrix_svd_stats_step_aggregate.csv").exists():
        print(f"Wrote: {out_dir / 'matrix_svd_stats_step_aggregate.csv'}")
        print(f"Wrote matrix-level evolution plots: matrix_<matrix>_<kind>_stats_evolution.png")


if __name__ == "__main__":
    main()
