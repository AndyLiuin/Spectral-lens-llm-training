from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def asha_rung_steps(max_steps: int, schedule: List[float]) -> List[int]:
    out: List[int] = []
    full_steps = max(1, int(max_steps))
    for item in schedule:
        val = float(item)
        if not np.isfinite(val) or val <= 0:
            continue
        if val <= 1.0:
            steps = int(np.ceil(full_steps * val))
        else:
            steps = int(round(val))
        out.append(min(full_steps, max(1, steps)))
    out.append(full_steps)
    return sorted(set(out))


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
    mins, maxs, pooled = [], [], []
    for df in dfs:
        x = pd.to_numeric(df.get("loss", pd.Series(dtype=float)), errors="coerce").dropna()
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


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = pd.Series(xs[mask]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(ys[mask]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def lr_candidates_for_batch(
    batch: int,
    *,
    base_batch: int,
    base_lr: float,
    lr_scaling: str,
    lr_multipliers: List[float],
    explicit_lr_list: List[float],
) -> List[float]:
    if explicit_lr_list:
        return sorted(set(float(x) for x in explicit_lr_list if x > 0))

    centers: List[float] = []
    scale = float(batch) / float(max(base_batch, 1))
    if lr_scaling in {"sqrt", "both"}:
        centers.append(float(base_lr) * float(np.sqrt(scale)))
    if lr_scaling in {"linear", "both"}:
        centers.append(float(base_lr) * scale)
    if lr_scaling == "none":
        centers.append(float(base_lr))
    if not centers:
        centers.append(float(base_lr))

    out: List[float] = []
    for center in centers:
        for mult in lr_multipliers:
            lr = float(center) * float(mult)
            if lr > 0:
                out.append(lr)
    return sorted(set(out))


def final_metric_value(metrics_df: pd.DataFrame, metric_name: str) -> float:
    if metrics_df.empty or metric_name not in metrics_df.columns:
        return float("nan")
    col = pd.to_numeric(metrics_df[metric_name], errors="coerce").dropna()
    return float(col.iloc[-1]) if not col.empty else float("nan")


def target_reached(state, metric_name: str, target_loss: float | None) -> bool:
    if target_loss is None or not np.isfinite(float(target_loss)):
        return False
    monitored = final_metric_value(state.metrics_df, metric_name)
    stop_reason = str(state.summary_row.get("stop_reason", ""))
    return bool(stop_reason == "target_loss" or (np.isfinite(monitored) and monitored <= float(target_loss)))


def selection_key(trial_row: dict, *, target_loss: float | None, target_metric_name: str) -> Tuple[float, ...]:
    val_loss = float(trial_row.get("final_val_loss", np.nan))
    train_loss = float(trial_row.get("final_train_loss", np.nan))
    test_loss = float(trial_row.get("test_loss", np.nan))
    tokens_seen = float(trial_row.get("tokens_seen", np.nan))
    final_step = float(trial_row.get("final_step", np.nan))
    monitored = val_loss if target_metric_name == "loss" else train_loss
    reached = bool(trial_row.get("hit_target", False))

    if target_loss is not None and np.isfinite(float(target_loss)):
        gap = abs(monitored - float(target_loss)) if np.isfinite(monitored) else float("inf")
        if reached:
            return (0.0, tokens_seen, monitored, test_loss, final_step)
        return (1.0, gap, monitored, test_loss, tokens_seen, final_step)

    return (
        val_loss if np.isfinite(val_loss) else float("inf"),
        test_loss if np.isfinite(test_loss) else float("inf"),
        tokens_seen if np.isfinite(tokens_seen) else float("inf"),
        final_step if np.isfinite(final_step) else float("inf"),
    )


def early_metric_row(df: pd.DataFrame, checkpoint: int) -> pd.Series | None:
    if df.empty or "step" not in df.columns:
        return None
    work = df.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    work = work.dropna(subset=["step"]).sort_values("step")
    if work.empty:
        return None
    eligible = work[work["step"] <= int(checkpoint)]
    if not eligible.empty:
        return eligible.iloc[-1]
    return work.iloc[0]


def build_run_config(
    *,
    args,
    settings: Dict[str, object],
    optimizer_kwargs: Dict[str, object],
    track: str,
    batch: int,
    lr: float,
    seed: int,
    max_steps: int,
    output_root: Path,
    save_checkpoints: bool,
    checkpoint_every: int,
    save_param_spectra: bool,
    grad_svd_samples: int,
    param_spectrum_paths: Tuple[str, ...] = (),
) -> RunConfig:
    return RunConfig(
        task=args.task,
        track=track,
        d=args.d,
        beta=args.beta,
        p=args.P,
        D=args.D,
        seq_len=args.seq_len,
        B=int(batch),
        lr=float(lr),
        seed=int(seed),
        max_steps=int(max_steps),
        log_every=args.log_every,
        eval_every=args.eval_every,
        measurement_every=args.measurement_every,
        target_loss=args.target_loss,
        target_loss_metric=args.target_loss_metric,
        target_loss_patience=args.target_loss_patience,
        target_loss_min_steps=args.target_loss_min_steps,
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
        variant=str(settings["variant"]),
        optimizer_name=str(settings["optimizer_name"]),
        window_size=int(settings["window_size"]),
        attention_scale=settings["attention_scale"],
        num_layers=int(settings["num_layers"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        lm_head_softcap=args.lm_head_softcap,
        measurement=MeasurementConfig(n_samples=1024, fixed_samples=True, trace_normalize=True),
        save_checkpoints=bool(save_checkpoints),
        checkpoint_every=int(checkpoint_every),
        save_param_spectra=bool(save_param_spectra),
        grad_svd_samples=int(grad_svd_samples),
        param_spectrum_paths=param_spectrum_paths,
        output_root=Path(output_root),
        **optimizer_kwargs,
    )


def build_trial_row_from_state(
    *,
    state,
    task: str,
    track: str,
    variant_index: int,
    variant_stage: str,
    variant_combo: str,
    target_loss: float | None,
    target_metric_name: str,
    search_mode: str,
    search_status: str,
    search_rung: int | None = None,
    search_total_rungs: int | None = None,
    search_budget_steps: int | None = None,
    search_budget_tokens: float | None = None,
    search_budget_frac: float | None = None,
    search_candidate_index: int | None = None,
    search_rank: int | None = None,
    search_promoted: bool | None = None,
) -> dict:
    final = state.metrics_df.sort_values("step").iloc[-1]
    row = {
        "task": task,
        "track": track,
        "variant_index": int(variant_index),
        "variant_stage": variant_stage,
        "variant_combo": variant_combo,
        "seed": int(state.config.seed),
        "B": int(state.config.B),
        "lr": float(state.config.lr),
        "final_step": int(state.summary_row.get("final_step", final.get("step", state.config.max_steps))),
        "final_val_loss": float(final.get("loss", np.nan)),
        "final_train_loss": float(final.get("train_loss", np.nan)),
        "test_loss": float(state.test_loss),
        "rankme": float(final.get("rankme", np.nan)),
        "alpha_head": float(final.get("alpha_head", np.nan)),
        "alpha_tail": float(final.get("alpha_tail", np.nan)),
        "grad_rankme": float(final.get("grad_rankme", np.nan)),
        "grad_alpha_head": float(final.get("grad_alpha_head", np.nan)),
        "grad_alpha_tail": float(final.get("grad_alpha_tail", np.nan)),
        "tokens_seen": float(final.get("tokens_seen", np.nan)),
        "train_time_s": float(final.get("train_time_s", np.nan)),
        "val_time_s": float(final.get("val_time_s", np.nan)),
        "checkpoint_time_s": float(final.get("checkpoint_time_s", np.nan)),
        "measurement_time_s": float(final.get("measurement_time_s", np.nan)),
        "stop_reason": state.summary_row.get("stop_reason", "max_steps"),
        "stopped_early": bool(state.summary_row.get("stopped_early", False)),
        "target_loss": float(target_loss) if target_loss is not None else np.nan,
        "target_loss_metric": "val" if target_metric_name == "loss" else "train",
        "hit_target": target_reached(state, target_metric_name, target_loss),
        "run_name": state.config.run_name(),
        "run_dir": str(state.run_dir),
        "search_mode": search_mode,
        "search_status": search_status,
        "search_rung": np.nan if search_rung is None else int(search_rung),
        "search_total_rungs": np.nan if search_total_rungs is None else int(search_total_rungs),
        "search_budget_steps": np.nan if search_budget_steps is None else int(search_budget_steps),
        "search_budget_tokens": np.nan if search_budget_tokens is None else float(search_budget_tokens),
        "search_budget_frac": np.nan if search_budget_frac is None else float(search_budget_frac),
        "search_candidate_index": np.nan if search_candidate_index is None else int(search_candidate_index),
        "search_rank": np.nan if search_rank is None else int(search_rank),
        "search_promoted": np.nan if search_promoted is None else bool(search_promoted),
    }
    return row


def response_rows_from_selected(selected_rows: pd.DataFrame, target_loss: float | None) -> pd.DataFrame:
    if selected_rows.empty:
        return pd.DataFrame()
    out = selected_rows.copy()
    if target_loss is not None and np.isfinite(float(target_loss)):
        out["hit_target"] = out["hit_target"].astype(bool)
        out["response_metric"] = "token_ratio_to_target"
        out["response_value"] = np.nan
        for (_, track, stage, seed), g in out.groupby(["task", "track", "variant_combo", "seed"]):
            idx = g.index[g["hit_target"]]
            if len(idx) == 0:
                continue
            denom = float(np.nanmin(pd.to_numeric(g.loc[idx, "tokens_seen"], errors="coerce")))
            if not np.isfinite(denom) or denom <= 0:
                continue
            out.loc[idx, "response_value"] = pd.to_numeric(out.loc[idx, "tokens_seen"], errors="coerce") / denom
        return out

    out["response_metric"] = "final_val_loss"
    out["response_value"] = pd.to_numeric(out["final_val_loss"], errors="coerce")
    return out


def build_variant_concat_summary_row(
    *,
    state,
    task: str,
    track: str,
    variant_stage: str,
    variant_index: int,
    variant_combo: str,
) -> dict:
    final = state.metrics_df.sort_values("step").iloc[-1]
    return {
        "task": task,
        "track": track,
        "variant_stage": variant_stage,
        "variant_index": int(variant_index),
        "variant_combo": variant_combo,
        "seed": int(state.config.seed),
        "d": int(state.config.d),
        "beta": float(state.config.beta),
        "P": int(state.config.p),
        "D": int(state.config.D),
        "B": int(state.config.B),
        "lr": float(state.config.lr),
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
        "val_time_s": float(final.get("val_time_s", np.nan)),
        "checkpoint_time_s": float(final.get("checkpoint_time_s", np.nan)),
        "measurement_time_s": float(final.get("measurement_time_s", np.nan)),
        "target_loss": float(state.config.target_loss) if state.config.target_loss is not None else np.nan,
        "target_loss_metric": state.config.target_loss_metric,
        "hit_target": target_reached(
            state,
            "loss" if str(state.config.target_loss_metric).strip().lower() == "val" else "train_loss",
            state.config.target_loss,
        ),
        "stop_reason": state.summary_row.get("stop_reason", "max_steps"),
        "stopped_early": bool(state.summary_row.get("stopped_early", False)),
        "final_step": int(state.summary_row.get("final_step", final["step"])),
        "test_loss": float(state.test_loss),
        "window_size": int(state.config.window_size),
        "attention_scale": state.config.attention_scale,
        "run_name": state.config.run_name(),
        "run_dir": str(state.run_dir),
    }


def build_variant_concat_matched_tables(
    *,
    tracks: Sequence[str],
    seeds: Sequence[int],
    batches: Sequence[int],
    stages: Sequence[Tuple[int, str, str]],
    registry: Dict[Tuple[str, str, int, int], dict],
    num_target_loss: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows: List[dict] = []
    baseline_combo = stages[0][2] if stages else "baseline"
    for track in tracks:
        for _, variant_stage, variant_combo in stages[1:]:
            for batch in batches:
                baseline_dfs: List[pd.DataFrame] = []
                variant_dfs: List[pd.DataFrame] = []
                for seed in seeds:
                    kb = (track, baseline_combo, int(seed), int(batch))
                    kv = (track, variant_combo, int(seed), int(batch))
                    if kb in registry and kv in registry:
                        baseline_dfs.append(registry[kb]["metrics_df"])
                        variant_dfs.append(registry[kv]["metrics_df"])
                if not baseline_dfs or not variant_dfs:
                    continue

                targets = common_loss_targets(baseline_dfs + variant_dfs, num_targets=num_target_loss)
                if len(targets) == 0:
                    continue

                for seed in seeds:
                    kb = (track, baseline_combo, int(seed), int(batch))
                    kv = (track, variant_combo, int(seed), int(batch))
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
                        sb = load_cov_spectrum(Path(str(registry[kb]["run_dir"])), int(rb.iloc[0]["step"]))
                        sv = load_cov_spectrum(Path(str(registry[kv]["run_dir"])), int(rv.iloc[0]["step"]))
                        matched_rows.append(
                            {
                                "track": track,
                                "variant_stage": variant_stage,
                                "variant_combo": variant_combo,
                                "seed": int(seed),
                                "B": int(batch),
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
    agg_df = (
        matched_df.groupby(["track", "B", "variant_stage", "variant_combo", "matched_loss_target"], as_index=False)
        .agg(
            js_div_mean=("js_div_baseline_vs_variant", "mean"),
            js_div_std=("js_div_baseline_vs_variant", "std"),
            n=("js_div_baseline_vs_variant", "size"),
        )
        if not matched_df.empty
        else pd.DataFrame(
            columns=["track", "B", "variant_stage", "variant_combo", "matched_loss_target", "js_div_mean", "js_div_std", "n"]
        )
    )
    return matched_df, agg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative-stage batch/LR sweep for toy models.")
    parser.add_argument("--task", type=str, default="mod_arith_lm", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--tracks", type=str, default="a")
    parser.add_argument("--variant-order", type=str, default="baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--D", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--B-list", type=str, default="32,128,512")
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--lr-multipliers", type=str, default="0.5,0.707,1.0,1.414,2.0")
    parser.add_argument("--lr-list", type=str, default="")
    parser.add_argument("--lr-scaling", type=str, default="both", choices=["sqrt", "linear", "both", "none"])
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--measurement-every", type=int, default=50)
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--target-loss-metric", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--target-loss-patience", type=int, default=1)
    parser.add_argument("--target-loss-min-steps", type=int, default=0)
    parser.add_argument("--num-target-loss", type=int, default=5)
    parser.add_argument("--early-checkpoints", type=str, default="200,400,800")
    parser.add_argument("--asha", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--asha-eta", type=int, default=2)
    parser.add_argument(
        "--asha-rungs",
        type=str,
        default="0.125,0.5,1.0",
        help="Comma-separated ASHA rung budgets as fractions of max_steps or absolute step counts.",
    )
    parser.add_argument("--materialize-selected-runs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selected-out-root", type=str, default="")
    parser.add_argument("--selected-ablation-name", type=str, default="variant_concat_ablation")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--save-param-spectra", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-svd-samples", type=int, default=32)
    parser.add_argument(
        "--param-spectrum-paths",
        type=str,
        default="",
        help="Optional comma-separated matrix paths for selected-run replay spectra.",
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
    parser.add_argument("--modarith-measurement-pooling", type=str, default="last", choices=["token", "last", "mean"])
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    tracks = parse_list(args.tracks)
    stages = cumulative_variant_combos(parse_list(args.variant_order))
    batches = parse_int_list(args.B_list)
    seeds = parse_int_list(args.seeds)
    lr_multipliers = parse_float_list(args.lr_multipliers)
    explicit_lr_list = parse_float_list(args.lr_list) if str(args.lr_list).strip() else []
    early_checkpoints = parse_int_list(args.early_checkpoints)
    asha_schedule = parse_float_list(args.asha_rungs) if str(args.asha_rungs).strip() else []
    asha_rungs = asha_rung_steps(args.max_steps, asha_schedule) if args.asha else []
    base_batch = batches[0] if batches else 1
    target_metric_name = "loss" if str(args.target_loss_metric).strip().lower() == "val" else "train_loss"
    optimizer_kwargs = optimizer_group_kwargs_from_args(args)
    search_output_root = Path(args.out_root)

    trial_rows: List[dict] = []
    registry: Dict[Tuple[str, str, int, int, float], dict] = {}
    selected_rows: List[dict] = []
    selected_registry: Dict[Tuple[str, str, int, int], dict] = {}

    for track in tracks:
        for variant_index, variant_stage, variant_combo in stages:
            settings = resolve_variant_settings(
                variant_combo,
                optimizer_name="adamw",
                num_layers=args.num_layers,
                window_size=args.window_size,
                attention_scale=args.attention_scale,
            )
            for batch in batches:
                lr_candidates = lr_candidates_for_batch(
                    batch,
                    base_batch=base_batch,
                    base_lr=args.base_lr,
                    lr_scaling=args.lr_scaling,
                    lr_multipliers=lr_multipliers,
                    explicit_lr_list=explicit_lr_list,
                )
                for seed in seeds:
                    if not args.asha:
                        for cand_idx, lr in enumerate(lr_candidates):
                            cfg = build_run_config(
                                args=args,
                                settings=settings,
                                optimizer_kwargs=optimizer_kwargs,
                                track=track,
                                batch=batch,
                                lr=lr,
                                seed=seed,
                                max_steps=args.max_steps,
                                output_root=search_output_root,
                                save_checkpoints=False,
                                checkpoint_every=0,
                                save_param_spectra=False,
                                grad_svd_samples=args.grad_svd_samples,
                            )
                            state = train_toy_run(
                                config=cfg,
                                ablation_name="concat_batch_regime_ablation",
                                write_summary=True,
                                save_spectra=True,
                                device=args.device,
                            )
                            trial_row = build_trial_row_from_state(
                                state=state,
                                task=args.task,
                                track=track,
                                variant_index=variant_index,
                                variant_stage=variant_stage,
                                variant_combo=variant_combo,
                                target_loss=args.target_loss,
                                target_metric_name=target_metric_name,
                                search_mode="grid",
                                search_status="complete",
                                search_budget_steps=args.max_steps,
                                search_budget_tokens=float(args.max_steps * batch * args.seq_len),
                                search_budget_frac=1.0,
                                search_candidate_index=cand_idx,
                                search_rank=cand_idx + 1,
                                search_promoted=True,
                            )
                            trial_rows.append(trial_row)
                            registry[(track, variant_combo, batch, seed, float(lr))] = {
                                "metrics_df": state.metrics_df,
                                "run_dir": state.run_dir,
                                "trial_row": trial_row,
                            }
                    else:
                        active = [{"lr": float(lr), "candidate_index": int(i)} for i, lr in enumerate(lr_candidates)]
                        finalists: List[dict] = []
                        total_rungs = len(asha_rungs)
                        for rung_idx, budget_steps in enumerate(asha_rungs, start=1):
                            if not active:
                                break
                            rung_items: List[dict] = []
                            for cand in active:
                                lr = float(cand["lr"])
                                cfg = build_run_config(
                                    args=args,
                                    settings=settings,
                                    optimizer_kwargs=optimizer_kwargs,
                                    track=track,
                                    batch=batch,
                                    lr=lr,
                                    seed=seed,
                                    max_steps=budget_steps,
                                    output_root=search_output_root,
                                    save_checkpoints=False,
                                    checkpoint_every=0,
                                    save_param_spectra=False,
                                    grad_svd_samples=args.grad_svd_samples,
                                )
                                state = train_toy_run(
                                    config=cfg,
                                    ablation_name="concat_batch_regime_ablation",
                                    write_summary=True,
                                    save_spectra=True,
                                    device=args.device,
                                )
                                trial_row = build_trial_row_from_state(
                                    state=state,
                                    task=args.task,
                                    track=track,
                                    variant_index=variant_index,
                                    variant_stage=variant_stage,
                                    variant_combo=variant_combo,
                                    target_loss=args.target_loss,
                                    target_metric_name=target_metric_name,
                                    search_mode="asha",
                                    search_status="pending",
                                    search_rung=rung_idx,
                                    search_total_rungs=total_rungs,
                                    search_budget_steps=budget_steps,
                                    search_budget_tokens=float(budget_steps * batch * args.seq_len),
                                    search_budget_frac=float(budget_steps) / float(max(args.max_steps, 1)),
                                    search_candidate_index=int(cand["candidate_index"]),
                                )
                                if bool(trial_row["hit_target"]) or rung_idx == total_rungs:
                                    trial_row["search_status"] = "hit_target" if bool(trial_row["hit_target"]) else "survivor"
                                    trial_row["search_promoted"] = True
                                    trial_rows.append(trial_row)
                                    finalists.append(
                                        {
                                            "metrics_df": state.metrics_df,
                                            "run_dir": state.run_dir,
                                            "trial_row": trial_row,
                                        }
                                    )
                                else:
                                    rung_items.append(
                                        {
                                            "candidate_index": int(cand["candidate_index"]),
                                            "lr": lr,
                                            "metrics_df": state.metrics_df,
                                            "run_dir": state.run_dir,
                                            "trial_row": trial_row,
                                        }
                                    )

                            if rung_idx == total_rungs or not rung_items:
                                active = []
                                continue

                            rung_items.sort(
                                key=lambda item: selection_key(
                                    item["trial_row"],
                                    target_loss=args.target_loss,
                                    target_metric_name=target_metric_name,
                                )
                            )
                            n_promote = max(1, len(rung_items) // max(1, int(args.asha_eta)))
                            next_active: List[dict] = []
                            for rank, item in enumerate(rung_items, start=1):
                                promoted = rank <= n_promote
                                item["trial_row"]["search_rank"] = rank
                                item["trial_row"]["search_promoted"] = promoted
                                item["trial_row"]["search_status"] = "promoted" if promoted else "pruned"
                                trial_rows.append(item["trial_row"])
                                if promoted:
                                    next_active.append(
                                        {
                                            "lr": float(item["lr"]),
                                            "candidate_index": int(item["candidate_index"]),
                                        }
                                    )
                            active = next_active

                        if finalists:
                            best = min(
                                finalists,
                                key=lambda item: selection_key(
                                    item["trial_row"],
                                    target_loss=args.target_loss,
                                    target_metric_name=target_metric_name,
                                ),
                            )
                            best_row = dict(best["trial_row"])
                            best_row["selection_mode"] = "target_loss" if args.target_loss is not None else "best_final_val"
                            selected_rows.append(best_row)
                            selected_registry[(track, variant_combo, int(batch), int(seed))] = {
                                "metrics_df": best["metrics_df"],
                                "run_dir": best["run_dir"],
                                "trial_row": best_row,
                            }

    trials_df = pd.DataFrame(trial_rows)

    if not args.asha and not trials_df.empty:
        group_cols = ["task", "track", "variant_index", "variant_stage", "variant_combo", "B", "seed"]
        for _, group in trials_df.groupby(group_cols):
            records = group.to_dict(orient="records")
            best = min(
                records,
                key=lambda row: selection_key(
                    row,
                    target_loss=args.target_loss,
                    target_metric_name=target_metric_name,
                ),
            )
            best = dict(best)
            best["selection_mode"] = "target_loss" if args.target_loss is not None else "best_final_val"
            selected_rows.append(best)
            selected_registry[(best["track"], best["variant_combo"], int(best["B"]), int(best["seed"]))] = registry[
                (best["track"], best["variant_combo"], int(best["B"]), int(best["seed"]), float(best["lr"]))
            ]

    selected_df = pd.DataFrame(selected_rows)

    selected_replay_rows: List[dict] = []
    selected_replay_registry: Dict[Tuple[str, str, int, int], dict] = {}
    selected_summary_df = pd.DataFrame()
    selected_matched_df = pd.DataFrame()
    selected_agg_df = pd.DataFrame()
    selected_out_dir = None
    if args.materialize_selected_runs and not selected_df.empty:
        selected_output_root = Path(args.selected_out_root) if str(args.selected_out_root).strip() else (Path(args.out_root) / "concat_batch_regime_selected_runs")
        selected_out_dir = selected_output_root / args.selected_ablation_name
        param_spectrum_paths = tuple(parse_list(args.param_spectrum_paths))
        for _, row in selected_df.sort_values(["track", "variant_index", "seed", "B"]).iterrows():
            track = str(row["track"])
            variant_combo = str(row["variant_combo"])
            seed = int(row["seed"])
            settings = resolve_variant_settings(
                variant_combo,
                optimizer_name="adamw",
                num_layers=args.num_layers,
                window_size=args.window_size,
                attention_scale=args.attention_scale,
            )
            cfg = build_run_config(
                args=args,
                settings=settings,
                optimizer_kwargs=optimizer_kwargs,
                track=track,
                batch=int(row["B"]),
                lr=float(row["lr"]),
                seed=seed,
                max_steps=args.max_steps,
                output_root=selected_output_root,
                save_checkpoints=args.save_checkpoints,
                checkpoint_every=args.checkpoint_every,
                save_param_spectra=args.save_param_spectra,
                grad_svd_samples=args.grad_svd_samples,
                param_spectrum_paths=param_spectrum_paths,
            )
            state = train_toy_run(
                config=cfg,
                ablation_name=args.selected_ablation_name,
                write_summary=True,
                save_spectra=True,
                device=args.device,
            )
            selected_replay_rows.append(
                build_variant_concat_summary_row(
                    state=state,
                    task=args.task,
                    track=track,
                    variant_stage=str(row["variant_stage"]),
                    variant_index=int(row["variant_index"]),
                    variant_combo=variant_combo,
                )
            )
            selected_replay_registry[(track, variant_combo, seed, int(row["B"]))] = {
                "metrics_df": state.metrics_df,
                "run_dir": state.run_dir,
                "variant_stage": str(row["variant_stage"]),
                "variant_index": int(row["variant_index"]),
            }

        selected_summary_df = pd.DataFrame(selected_replay_rows)
        selected_matched_df, selected_agg_df = build_variant_concat_matched_tables(
            tracks=tracks,
            seeds=seeds,
            batches=batches,
            stages=stages,
            registry=selected_replay_registry,
            num_target_loss=args.num_target_loss,
        )

    matched_row_list: List[dict] = []
    pair_row_list: List[dict] = []
    if not selected_df.empty:
        for (track, variant_combo, seed), group in selected_df.groupby(["track", "variant_combo", "seed"]):
            selected_states = [selected_registry[(track, variant_combo, int(r["B"]), int(seed))] for _, r in group.iterrows()]
            dfs = [x["metrics_df"] for x in selected_states]
            targets = common_loss_targets(dfs, num_targets=args.num_target_loss)
            if len(targets) == 0:
                continue

            matched_tables: Dict[int, pd.DataFrame] = {}
            for _, row in group.iterrows():
                key = (track, variant_combo, int(row["B"]), int(seed))
                st = selected_registry[key]
                matched = matched_loss_rows(st["metrics_df"], targets)
                if matched.empty:
                    continue
                matched = matched.copy()
                matched["task"] = row["task"]
                matched["track"] = track
                matched["variant_combo"] = variant_combo
                matched["variant_stage"] = row["variant_stage"]
                matched["variant_index"] = int(row["variant_index"])
                matched["seed"] = int(seed)
                matched["B"] = int(row["B"])
                matched["lr"] = float(row["lr"])
                matched["run_dir"] = str(st["run_dir"])
                matched["hit_target"] = bool(row["hit_target"])
                matched_row_list.extend(matched.to_dict(orient="records"))
                matched_tables[int(row["B"])] = matched

            for target in targets:
                for left_b, right_b in combinations(sorted(matched_tables.keys()), 2):
                    left = matched_tables[left_b]
                    right = matched_tables[right_b]
                    lrow = left[np.isclose(left["matched_loss_target"], target)]
                    rrow = right[np.isclose(right["matched_loss_target"], target)]
                    if lrow.empty or rrow.empty:
                        continue
                    litem = lrow.iloc[0]
                    ritem = rrow.iloc[0]
                    s1 = load_cov_spectrum(Path(str(litem["run_dir"])), int(litem["step"]))
                    s2 = load_cov_spectrum(Path(str(ritem["run_dir"])), int(ritem["step"]))
                    pair_row_list.append(
                        {
                            "task": litem["task"],
                            "track": track,
                            "variant_combo": variant_combo,
                            "variant_stage": litem["variant_stage"],
                            "variant_index": int(litem["variant_index"]),
                            "seed": int(seed),
                            "matched_loss_target": float(target),
                            "left_B": int(left_b),
                            "right_B": int(right_b),
                            "left_lr": float(litem["lr"]),
                            "right_lr": float(ritem["lr"]),
                            "left_step": int(litem["step"]),
                            "right_step": int(ritem["step"]),
                            "left_match_error": float(litem.get("matched_loss_error", np.nan)),
                            "right_match_error": float(ritem.get("matched_loss_error", np.nan)),
                            "js_div_cov": js_divergence(s1, s2),
                        }
                    )

    early_rows: List[dict] = []
    selected_with_response = response_rows_from_selected(selected_df, target_loss=args.target_loss)
    predictors = [
        "rankme",
        "alpha_head",
        "alpha_tail",
        "grad_rankme",
        "grad_alpha_head",
        "grad_alpha_tail",
        "act_grad_head_gap",
        "act_grad_tail_gap",
    ]
    if not selected_with_response.empty:
        for (track, variant_combo), group in selected_with_response.groupby(["track", "variant_combo"]):
            for checkpoint in early_checkpoints:
                point_rows: List[dict] = []
                for _, row in group.iterrows():
                    key = (track, variant_combo, int(row["B"]), int(row["seed"]))
                    state = selected_registry.get(key)
                    if state is None:
                        continue
                    early = early_metric_row(state["metrics_df"], checkpoint)
                    if early is None:
                        continue
                    item = {
                        "task": row["task"],
                        "track": track,
                        "variant_combo": variant_combo,
                        "variant_stage": row["variant_stage"],
                        "variant_index": int(row["variant_index"]),
                        "seed": int(row["seed"]),
                        "B": int(row["B"]),
                        "checkpoint": int(checkpoint),
                        "response_metric": row["response_metric"],
                        "response_value": float(row["response_value"]),
                        "rankme": float(early.get("rankme", np.nan)),
                        "alpha_head": float(early.get("alpha_head", np.nan)),
                        "alpha_tail": float(early.get("alpha_tail", np.nan)),
                        "grad_rankme": float(early.get("grad_rankme", np.nan)),
                        "grad_alpha_head": float(early.get("grad_alpha_head", np.nan)),
                        "grad_alpha_tail": float(early.get("grad_alpha_tail", np.nan)),
                    }
                    item["act_grad_head_gap"] = item["alpha_head"] - item["grad_alpha_head"]
                    item["act_grad_tail_gap"] = item["alpha_tail"] - item["grad_alpha_tail"]
                    point_rows.append(item)

                if not point_rows:
                    continue
                point_df = pd.DataFrame(point_rows)
                for predictor in predictors:
                    early_rows.append(
                        {
                            "task": point_df["task"].iloc[0],
                            "track": track,
                            "variant_combo": variant_combo,
                            "variant_stage": point_df["variant_stage"].iloc[0],
                            "variant_index": int(point_df["variant_index"].iloc[0]),
                            "checkpoint": int(checkpoint),
                            "predictor": predictor,
                            "response_metric": point_df["response_metric"].iloc[0],
                            "spearman_r": safe_spearman(point_df[predictor].to_numpy(dtype=float), point_df["response_value"].to_numpy(dtype=float)),
                            "n": int(len(point_df)),
                        }
                    )

    out_dir = Path(args.out_root) / "concat_batch_regime_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_df.to_csv(out_dir / "concat_batch_regime_trials.csv", index=False)
    selected_df.to_csv(out_dir / "concat_batch_regime_selected_lrs.csv", index=False)
    selected_df.to_csv(out_dir / "constant_loss_stage_summary.csv", index=False)
    pd.DataFrame(matched_row_list).to_csv(out_dir / "concat_batch_regime_matched_loss_rows.csv", index=False)
    pd.DataFrame(pair_row_list).to_csv(out_dir / "concat_batch_regime_matched_pairs.csv", index=False)
    pd.DataFrame(early_rows).to_csv(out_dir / "concat_batch_regime_early_prediction.csv", index=False)

    print(f"Wrote: {out_dir / 'concat_batch_regime_trials.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_selected_lrs.csv'}")
    print(f"Wrote: {out_dir / 'constant_loss_stage_summary.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_matched_loss_rows.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_matched_pairs.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_early_prediction.csv'}")

    if selected_out_dir is not None:
        selected_out_dir.mkdir(parents=True, exist_ok=True)
        selected_summary_df.to_csv(selected_out_dir / "variant_concat_ablation_summary.csv", index=False)
        selected_matched_df.to_csv(selected_out_dir / "variant_concat_ablation_matched_pairs.csv", index=False)
        selected_agg_df.to_csv(selected_out_dir / "variant_concat_ablation_matched_aggregate.csv", index=False)
        replay_manifest = selected_df.copy()
        replay_lookup = {
            (str(r["track"]), str(r["variant_combo"]), int(r["seed"]), int(r["B"])): str(r["run_dir"])
            for _, r in selected_summary_df.iterrows()
        }
        replay_manifest["selected_run_dir"] = [
            replay_lookup.get((str(r["track"]), str(r["variant_combo"]), int(r["seed"]), int(r["B"])), "")
            for _, r in replay_manifest.iterrows()
        ]
        replay_manifest.to_csv(out_dir / "concat_batch_regime_selected_replay_manifest.csv", index=False)
        print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_summary.csv'}")
        print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_matched_pairs.csv'}")
        print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_matched_aggregate.csv'}")
        print(f"Wrote: {out_dir / 'concat_batch_regime_selected_replay_manifest.csv'}")


if __name__ == "__main__":
    main()
