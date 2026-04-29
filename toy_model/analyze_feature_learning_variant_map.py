from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .runner import matched_loss_rows
except ImportError:
    from runner import matched_loss_rows


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _alignment_columns(feature_df: pd.DataFrame, alignment_mode: str) -> Tuple[str, str]:
    mode = str(alignment_mode).strip().lower()
    if mode == "progress" and "checkpoint_align_key" in feature_df.columns and "checkpoint_progress_pct" in feature_df.columns:
        return "checkpoint_align_key", "checkpoint_progress_pct"
    return "checkpoint", "checkpoint"


def parse_band_list(text: str) -> List[str]:
    out: List[str] = []
    for chunk in str(text).split(","):
        s = chunk.strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"Invalid band '{s}'. Expected c0:o0 format.")
        out.append(s)
    return out


def _trajectory_cols(df: pd.DataFrame) -> List[str]:
    cols = ["track", "seed"]
    if "B" in df.columns:
        cols.append("B")
    return [c for c in cols if c in df.columns]


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
    return np.unique(np.quantile(pooled_arr, qs).astype(float))


def load_cov_spectrum(run_dir: Path, step: int) -> np.ndarray:
    path = run_dir / "spectra" / f"cov_spectrum_step_{int(step):06d}.npy"
    if not path.exists():
        return np.array([])
    return np.load(path)


def _safe_ratio_gain(prev_value: float, curr_value: float) -> float:
    if prev_value > 0 and curr_value > 0:
        return float(prev_value / curr_value - 1.0)
    return float("nan")


def _safe_log_ratio(prev_value: float, curr_value: float) -> float:
    if prev_value > 0 and curr_value > 0:
        return float(np.log(prev_value / curr_value))
    return float("nan")


def _summary_hit_target(row: pd.Series) -> bool:
    if "hit_target" in row and pd.notna(row.get("hit_target", np.nan)):
        return bool(row.get("hit_target"))
    stop_reason = str(row.get("stop_reason", "")).strip().lower()
    if stop_reason == "target_loss":
        return True
    target_loss = pd.to_numeric(pd.Series([row.get("target_loss", np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(target_loss):
        return False
    metric_mode = str(row.get("target_loss_metric", "val")).strip().lower()
    if metric_mode == "train":
        metric_candidates = ["final_train_loss", "train_loss"]
    else:
        metric_candidates = ["final_val_loss", "loss"]
    for col in metric_candidates:
        value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(value):
            return bool(value <= float(target_loss))
    return False


def _resolve_run_dir(row: pd.Series, ablation_dir: Path, prefix: str = "") -> Optional[Path]:
    candidates: List[Path] = []
    run_name_col = f"{prefix}run_name"
    run_dir_col = f"{prefix}run_dir"
    run_name = row.get(run_name_col, "")
    if isinstance(run_name, str) and run_name.strip():
        candidates.append(ablation_dir / run_name.strip())
    run_dir = row.get(run_dir_col, "")
    if isinstance(run_dir, str) and run_dir.strip():
        p = Path(run_dir.strip())
        candidates.append(p if p.is_absolute() else (Path.cwd() / p))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_metrics_df(run_dir: Optional[Path], cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    key = str(run_dir.resolve())
    if key not in cache:
        path = run_dir / "metrics_over_time.csv"
        if path.exists():
            try:
                cache[key] = pd.read_csv(path)
            except Exception:
                cache[key] = pd.DataFrame()
        else:
            cache[key] = pd.DataFrame()
    return cache[key]


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


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots.append(float(np.mean(arr[idx])))
    q = np.quantile(np.asarray(boots, dtype=np.float64), [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def sign_flip_pvalue(values: Sequence[float], n_perm: int = 5000, seed: int = 0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    obs = abs(float(np.mean(arr)))
    rng = np.random.default_rng(seed)
    ge = 1
    total = 1
    for _ in range(int(n_perm)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=arr.size, replace=True)
        stat = abs(float(np.mean(arr * signs)))
        ge += int(stat >= obs)
        total += 1
    return float(ge / total)


def zscore_series(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
    return (x - mu) / sd


def _sustained_positive_onset(checkpoints: np.ndarray, values: np.ndarray, sustain_frac: float = 0.8) -> float:
    if checkpoints.size == 0 or values.size == 0:
        return float("nan")
    for i in range(len(values)):
        if not np.isfinite(values[i]) or values[i] <= 0.0:
            continue
        tail = values[i:]
        ok = np.isfinite(tail)
        if ok.sum() == 0:
            continue
        frac = float((tail[ok] > 0.0).mean())
        if frac >= float(sustain_frac):
            return float(checkpoints[i])
    return float("nan")


def _early_slope(checkpoints: np.ndarray, values: np.ndarray, max_points: int = 4) -> float:
    mask = np.isfinite(checkpoints) & np.isfinite(values)
    xs = checkpoints[mask].astype(np.float64)
    ys = values[mask].astype(np.float64)
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = xs[order][: max(int(max_points), 2)]
    ys = ys[order][: max(int(max_points), 2)]
    if xs.size < 2 or float(np.std(xs)) < 1e-12:
        return float("nan")
    slope, _ = np.polyfit(xs, ys, deg=1)
    return float(slope)


def build_transition_endpoints(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    traj_cols = _trajectory_cols(summary_df)
    for traj_key, g in summary_df.groupby(traj_cols):
        g = g.sort_values("variant_index").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            curr = g.iloc[i]
            prev_thr = float(prev["tokens_seen"] / prev["train_time_s"]) if prev["train_time_s"] > 0 else np.nan
            curr_thr = float(curr["tokens_seen"] / curr["train_time_s"]) if curr["train_time_s"] > 0 else np.nan
            prev_hit_target = _summary_hit_target(prev)
            curr_hit_target = _summary_hit_target(curr)
            rows.append(
                {
                    "track": prev["track"],
                    "seed": int(prev["seed"]),
                    "B": int(prev["B"]) if "B" in traj_cols else np.nan,
                    "prev_variant_index": int(prev["variant_index"]),
                    "variant_index": int(curr["variant_index"]),
                    "prev_variant_stage": prev["variant_stage"],
                    "variant_stage": curr["variant_stage"],
                    "prev_variant_combo": prev["variant_combo"],
                    "variant_combo": curr["variant_combo"],
                    "transition": f"{prev['variant_stage']}->{curr['variant_stage']}",
                    "prev_run_name": prev.get("run_name", ""),
                    "run_name": curr.get("run_name", ""),
                    "prev_run_dir": prev.get("run_dir", ""),
                    "run_dir": curr.get("run_dir", ""),
                    "prev_test_loss": float(prev["test_loss"]),
                    "test_loss": float(curr["test_loss"]),
                    "prev_tokens_seen": float(prev.get("tokens_seen", np.nan)),
                    "tokens_seen": float(curr.get("tokens_seen", np.nan)),
                    "prev_train_time_s": float(prev.get("train_time_s", np.nan)),
                    "train_time_s": float(curr.get("train_time_s", np.nan)),
                    "prev_throughput": prev_thr,
                    "throughput": curr_thr,
                    "tok_gain": _safe_ratio_gain(float(prev["test_loss"]), float(curr["test_loss"])),
                    "thr_gain": _safe_ratio_gain(curr_thr, prev_thr),
                    "time_gain": _safe_ratio_gain(float(prev["train_time_s"]), float(curr["train_time_s"])),
                    "log_tok_gain": _safe_log_ratio(float(prev["test_loss"]), float(curr["test_loss"])),
                    "log_thr_gain": _safe_log_ratio(curr_thr, prev_thr),
                    "prev_hit_target": bool(prev_hit_target),
                    "hit_target": bool(curr_hit_target),
                    "both_hit_target": bool(prev_hit_target and curr_hit_target),
                    "constant_loss_tok_gain": _safe_ratio_gain(float(prev.get("tokens_seen", np.nan)), float(curr.get("tokens_seen", np.nan)))
                    if prev_hit_target and curr_hit_target
                    else np.nan,
                    "constant_loss_time_gain": _safe_ratio_gain(float(prev.get("train_time_s", np.nan)), float(curr.get("train_time_s", np.nan)))
                    if prev_hit_target and curr_hit_target
                    else np.nan,
                    "log_constant_loss_tok_gain": _safe_log_ratio(float(prev.get("tokens_seen", np.nan)), float(curr.get("tokens_seen", np.nan)))
                    if prev_hit_target and curr_hit_target
                    else np.nan,
                    "log_constant_loss_time_gain": _safe_log_ratio(float(prev.get("train_time_s", np.nan)), float(curr.get("train_time_s", np.nan)))
                    if prev_hit_target and curr_hit_target
                    else np.nan,
                    "d_rankme_final": float(curr["rankme"] - prev["rankme"]),
                    "d_alpha_head_final": float(curr["alpha_head"] - prev["alpha_head"]),
                    "d_alpha_tail_final": float(curr["alpha_tail"] - prev["alpha_tail"]),
                    "d_grad_rankme_final": float(curr["grad_rankme"] - prev["grad_rankme"]),
                    "d_grad_alpha_head_final": float(curr["grad_alpha_head"] - prev["grad_alpha_head"]),
                    "d_grad_alpha_tail_final": float(curr["grad_alpha_tail"] - prev["grad_alpha_tail"]),
                }
            )
    return pd.DataFrame(rows)


def augment_loss_proximal_transition_metrics(
    endpoints: pd.DataFrame,
    *,
    ablation_dir: Path,
    num_targets: int,
) -> pd.DataFrame:
    if endpoints.empty:
        return endpoints.copy()

    metrics_cache: Dict[str, pd.DataFrame] = {}
    spectrum_cache: Dict[Tuple[str, int], np.ndarray] = {}
    rows: List[dict] = []
    join_cols = _trajectory_cols(endpoints) + [
        "prev_variant_index",
        "variant_index",
        "prev_variant_stage",
        "variant_stage",
        "transition",
    ]

    for _, tr in endpoints.iterrows():
        prev_run_dir = _resolve_run_dir(tr, ablation_dir, prefix="prev_")
        curr_run_dir = _resolve_run_dir(tr, ablation_dir, prefix="")
        prev_metrics = _load_metrics_df(prev_run_dir, cache=metrics_cache)
        curr_metrics = _load_metrics_df(curr_run_dir, cache=metrics_cache)
        out = {c: tr[c] for c in join_cols}
        out.update(
            {
                "loss_proximal_tok_gain_mean": np.nan,
                "loss_proximal_time_gain_mean": np.nan,
                "loss_proximal_js_div_mean": np.nan,
                "loss_proximal_loss_gap_mean": np.nan,
                "loss_proximal_n_targets": 0,
            }
        )
        if prev_metrics.empty or curr_metrics.empty:
            rows.append(out)
            continue

        targets = common_loss_targets([prev_metrics, curr_metrics], num_targets=num_targets)
        if len(targets) == 0:
            rows.append(out)
            continue

        prev_matched = matched_loss_rows(prev_metrics, targets)
        curr_matched = matched_loss_rows(curr_metrics, targets)
        if prev_matched.empty or curr_matched.empty:
            rows.append(out)
            continue

        tok_gains: List[float] = []
        time_gains: List[float] = []
        js_vals: List[float] = []
        loss_gaps: List[float] = []
        for target in targets:
            prev_row = prev_matched[np.isclose(prev_matched["matched_loss_target"], target)]
            curr_row = curr_matched[np.isclose(curr_matched["matched_loss_target"], target)]
            if prev_row.empty or curr_row.empty:
                continue
            prow = prev_row.iloc[0]
            crow = curr_row.iloc[0]
            tok_gains.append(_safe_ratio_gain(float(prow.get("tokens_seen", np.nan)), float(crow.get("tokens_seen", np.nan))))
            time_gains.append(_safe_ratio_gain(float(prow.get("train_time_s", np.nan)), float(crow.get("train_time_s", np.nan))))
            loss_gaps.append(float(abs(float(prow.get("loss", np.nan)) - float(crow.get("loss", np.nan)))))

            if prev_run_dir is None or curr_run_dir is None:
                continue
            prev_spec_key = (str(prev_run_dir.resolve()), int(prow["step"]))
            curr_spec_key = (str(curr_run_dir.resolve()), int(crow["step"]))
            if prev_spec_key not in spectrum_cache:
                spectrum_cache[prev_spec_key] = load_cov_spectrum(prev_run_dir, int(prow["step"]))
            if curr_spec_key not in spectrum_cache:
                spectrum_cache[curr_spec_key] = load_cov_spectrum(curr_run_dir, int(crow["step"]))
            js_vals.append(js_divergence(spectrum_cache[prev_spec_key], spectrum_cache[curr_spec_key]))

        tok_arr = np.asarray(tok_gains, dtype=np.float64)
        time_arr = np.asarray(time_gains, dtype=np.float64)
        js_arr = np.asarray(js_vals, dtype=np.float64)
        gap_arr = np.asarray(loss_gaps, dtype=np.float64)
        out["loss_proximal_tok_gain_mean"] = float(np.nanmean(tok_arr)) if np.isfinite(tok_arr).any() else np.nan
        out["loss_proximal_time_gain_mean"] = float(np.nanmean(time_arr)) if np.isfinite(time_arr).any() else np.nan
        out["loss_proximal_js_div_mean"] = float(np.nanmean(js_arr)) if np.isfinite(js_arr).any() else np.nan
        out["loss_proximal_loss_gap_mean"] = float(np.nanmean(gap_arr)) if np.isfinite(gap_arr).any() else np.nan
        out["loss_proximal_n_targets"] = int(np.isfinite(gap_arr).sum())
        rows.append(out)

    loss_prox_df = pd.DataFrame(rows)
    return endpoints.merge(loss_prox_df, on=join_cols, how="left")


def load_feature_tables(feature_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s_path = feature_dir / "feature_learning_summary.csv"
    c_path = feature_dir / "feature_causal_effects.csv"
    p_path = feature_dir / "feature_pca_summary.csv"
    if not s_path.exists() or not c_path.exists() or not p_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = pd.read_csv(s_path)
    causal = pd.read_csv(c_path)
    pca = pd.read_csv(p_path)
    return summary, causal, pca


def choose_main_probe_type(summary: pd.DataFrame, causal: pd.DataFrame) -> str:
    for df in (summary, causal):
        if df.empty or "probe_type" not in df.columns:
            continue
        vals = {str(x).strip() for x in df["probe_type"].dropna().unique()}
        if "matched_band" in vals:
            return "matched_band"
        if "clean_band" in vals:
            return "clean_band"
    return "clean_band"


def _preferred_metric_series(df: pd.DataFrame, hidden_col: str, embed_col: str) -> pd.Series:
    hidden = pd.to_numeric(df.get(hidden_col, np.nan), errors="coerce")
    embed = pd.to_numeric(df.get(embed_col, np.nan), errors="coerce")
    out = hidden.copy()
    mask = ~np.isfinite(out.to_numpy(dtype=float))
    out.loc[mask] = embed.loc[mask]
    return out


def build_feature_merged(summary: pd.DataFrame, causal: pd.DataFrame, pca: pd.DataFrame) -> pd.DataFrame:
    key = ["track", "variant_stage", "variant_index", "variant_combo", "seed"]
    if "B" in summary.columns and "B" in causal.columns:
        key.append("B")
    key.extend(["checkpoint", "probe_band", "probe_type"])
    pca_key = [c for c in key if c in pca.columns]

    if pca.empty or "pc_index" not in pca.columns:
        pca_pc1 = pd.DataFrame(columns=pca_key + ["pca_peak_mass_pc1", "pca_dominant_freq_pc1"])
    else:
        pca_pc1 = (
            pca[pca["pc_index"] == 1][pca_key + ["pca_peak_mass", "dominant_freq"]]
            .rename(columns={"pca_peak_mass": "pca_peak_mass_pc1", "dominant_freq": "pca_dominant_freq_pc1"})
            .copy()
        )

    merged = summary.merge(causal, on=key, how="left", suffixes=("", "_causal"))
    merged = merged.merge(pca_pc1, on=pca_key, how="left")

    # The summary table already carries PC1 peak mass, so merging the explicit
    # PCA table can create x/y suffixed duplicates. Normalize back to one name.
    if "pca_peak_mass_pc1" not in merged.columns:
        lhs = pd.to_numeric(merged.get("pca_peak_mass_pc1_x", np.nan), errors="coerce")
        rhs = pd.to_numeric(merged.get("pca_peak_mass_pc1_y", np.nan), errors="coerce")
        merged["pca_peak_mass_pc1"] = lhs
        mask = ~np.isfinite(merged["pca_peak_mass_pc1"].to_numpy(dtype=float))
        merged.loc[mask, "pca_peak_mass_pc1"] = rhs.loc[mask]

    for c in [
        "H_peak",
        "H_gini",
        "E_peak",
        "Emb_peak",
        "keep_key_loss",
        "drop_key_loss",
        "keep_ctrl_loss",
        "drop_ctrl_loss",
        "delta_keep",
        "delta_drop",
        "delta_keep_ctrl",
        "delta_drop_ctrl",
        "delta_keep_vs_ctrl",
        "delta_drop_vs_ctrl",
        "hidden_delta_keep",
        "hidden_delta_drop",
        "hidden_delta_keep_ctrl",
        "hidden_delta_drop_ctrl",
        "hidden_delta_keep_vs_ctrl",
        "hidden_delta_drop_vs_ctrl",
        "pca_peak_mass_pc1",
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if "dominant_freq" not in merged.columns and "pca_dominant_freq_pc1" in merged.columns:
        merged["dominant_freq"] = merged["pca_dominant_freq_pc1"]

    merged["causal_delta_keep"] = _preferred_metric_series(merged, "hidden_delta_keep", "delta_keep")
    merged["causal_delta_drop"] = _preferred_metric_series(merged, "hidden_delta_drop", "delta_drop")
    merged["causal_delta_keep_ctrl"] = _preferred_metric_series(merged, "hidden_delta_keep_ctrl", "delta_keep_ctrl")
    merged["causal_delta_drop_ctrl"] = _preferred_metric_series(merged, "hidden_delta_drop_ctrl", "delta_drop_ctrl")
    merged["causal_delta_keep_vs_ctrl"] = _preferred_metric_series(merged, "hidden_delta_keep_vs_ctrl", "delta_keep_vs_ctrl")
    merged["causal_delta_drop_vs_ctrl"] = _preferred_metric_series(merged, "hidden_delta_drop_vs_ctrl", "delta_drop_vs_ctrl")
    has_hidden = np.isfinite(pd.to_numeric(merged.get("hidden_delta_keep", np.nan), errors="coerce").to_numpy(dtype=float))
    merged["causal_metric_source"] = np.where(has_hidden, "hidden_state", "embedding")
    merged["drop_advantage"] = merged["causal_delta_drop_ctrl"] - merged["causal_delta_drop"]
    merged["keep_advantage"] = merged["causal_delta_keep"] - merged["causal_delta_keep_ctrl"]
    return merged


def build_transition_feature_map(
    endpoints: pd.DataFrame,
    feature_df: pd.DataFrame,
    checkpoints: Sequence[int],
    bands: Sequence[str],
    alignment_mode: str = "step",
) -> pd.DataFrame:
    if endpoints.empty:
        return pd.DataFrame()

    if feature_df.empty:
        out = endpoints.copy()
        out["checkpoint"] = np.nan
        out["probe_band"] = ""
        for c in [
            "H_peak",
            "E_peak",
            "Emb_peak",
            "keep_key_loss",
            "drop_key_loss",
            "keep_ctrl_loss",
            "drop_ctrl_loss",
            "delta_keep",
            "delta_drop",
            "causal_delta_keep_vs_ctrl",
            "causal_delta_drop_vs_ctrl",
            "dominant_freq",
            "pca_peak_mass_pc1",
            "delta_H_peak",
            "delta_drop_advantage",
            "delta_keep_advantage",
            "delta_pca_peak_mass_pc1",
        ]:
            out[c] = np.nan
        return out

    feats = feature_df.copy()
    align_key_col, align_value_col = _alignment_columns(feats, alignment_mode)
    if checkpoints:
        feats = feats[pd.to_numeric(feats[align_value_col], errors="coerce").isin(list(checkpoints))].copy()
    if bands:
        feats = feats[feats["probe_band"].isin(list(bands))].copy()

    if feats.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    traj_cols = [c for c in _trajectory_cols(endpoints if not endpoints.empty else feats) if c in feats.columns]
    key_cols = traj_cols + ["variant_stage", align_key_col, "probe_band", "probe_type"]
    by_key = feats.set_index(key_cols).sort_index()

    for _, tr in endpoints.iterrows():
        traj_vals = [tr[c] for c in traj_cols]
        prev_stage = tr["prev_variant_stage"]
        curr_stage = tr["variant_stage"]

        curr_mask = feats["variant_stage"] == curr_stage
        for c, v in zip(traj_cols, traj_vals):
            curr_mask = curr_mask & (feats[c] == v)
        curr_rows = feats[curr_mask]
        if curr_rows.empty:
            continue

        for _, curr in curr_rows.iterrows():
            idx_prev = (*traj_vals, prev_stage, curr[align_key_col], curr["probe_band"], curr["probe_type"])
            if idx_prev not in by_key.index:
                continue
            prev = by_key.loc[idx_prev]
            if isinstance(prev, pd.DataFrame):
                prev = prev.iloc[0]

            row = tr.to_dict()
            row.update(
                {
                    "checkpoint": int(pd.to_numeric(curr.get(align_value_col, np.nan), errors="coerce"))
                    if pd.notna(pd.to_numeric(curr.get(align_value_col, np.nan), errors="coerce"))
                    else np.nan,
                    "checkpoint_step": int(curr["checkpoint"]),
                    "alignment_mode": str(alignment_mode).strip().lower(),
                    "alignment_key": curr.get(align_key_col, ""),
                    "checkpoint_progress_pct": float(pd.to_numeric(curr.get("checkpoint_progress_pct", np.nan), errors="coerce")),
                    "checkpoint_progress_frac": float(pd.to_numeric(curr.get("checkpoint_progress_frac", np.nan), errors="coerce")),
                    "probe_band": curr["probe_band"],
                    "probe_type": curr.get("probe_type", ""),
                    "H_peak": float(curr.get("H_peak", np.nan)),
                    "E_peak": float(curr.get("E_peak", np.nan)),
                    "Emb_peak": float(curr.get("Emb_peak", np.nan)),
                    "keep_key_loss": float(curr.get("keep_key_loss", np.nan)),
                    "drop_key_loss": float(curr.get("drop_key_loss", np.nan)),
                    "keep_ctrl_loss": float(curr.get("keep_ctrl_loss", np.nan)),
                    "drop_ctrl_loss": float(curr.get("drop_ctrl_loss", np.nan)),
                    "delta_keep": float(curr.get("delta_keep", np.nan)),
                    "delta_drop": float(curr.get("delta_drop", np.nan)),
                    "hidden_delta_keep": float(curr.get("hidden_delta_keep", np.nan)),
                    "hidden_delta_drop": float(curr.get("hidden_delta_drop", np.nan)),
                    "hidden_delta_keep_ctrl": float(curr.get("hidden_delta_keep_ctrl", np.nan)),
                    "hidden_delta_drop_ctrl": float(curr.get("hidden_delta_drop_ctrl", np.nan)),
                    "causal_delta_keep": float(curr.get("causal_delta_keep", np.nan)),
                    "causal_delta_drop": float(curr.get("causal_delta_drop", np.nan)),
                    "causal_delta_keep_ctrl": float(curr.get("causal_delta_keep_ctrl", np.nan)),
                    "causal_delta_drop_ctrl": float(curr.get("causal_delta_drop_ctrl", np.nan)),
                    "causal_delta_keep_vs_ctrl": float(curr.get("causal_delta_keep_vs_ctrl", np.nan)),
                    "causal_delta_drop_vs_ctrl": float(curr.get("causal_delta_drop_vs_ctrl", np.nan)),
                    "causal_metric_source": curr.get("causal_metric_source", ""),
                    "dominant_freq": float(curr.get("dominant_freq", np.nan)),
                    "pca_peak_mass_pc1": float(curr.get("pca_peak_mass_pc1", np.nan)),
                    "prev_H_peak": float(prev.get("H_peak", np.nan)),
                    "prev_E_peak": float(prev.get("E_peak", np.nan)),
                    "prev_Emb_peak": float(prev.get("Emb_peak", np.nan)),
                    "prev_delta_keep": float(prev.get("delta_keep", np.nan)),
                    "prev_delta_drop": float(prev.get("delta_drop", np.nan)),
                    "prev_hidden_delta_keep": float(prev.get("hidden_delta_keep", np.nan)),
                    "prev_hidden_delta_drop": float(prev.get("hidden_delta_drop", np.nan)),
                    "prev_causal_delta_keep": float(prev.get("causal_delta_keep", np.nan)),
                    "prev_causal_delta_drop": float(prev.get("causal_delta_drop", np.nan)),
                    "prev_causal_delta_keep_vs_ctrl": float(prev.get("causal_delta_keep_vs_ctrl", np.nan)),
                    "prev_causal_delta_drop_vs_ctrl": float(prev.get("causal_delta_drop_vs_ctrl", np.nan)),
                    "prev_drop_advantage": float(prev.get("drop_advantage", np.nan)),
                    "prev_keep_advantage": float(prev.get("keep_advantage", np.nan)),
                    "prev_pca_peak_mass_pc1": float(prev.get("pca_peak_mass_pc1", np.nan)),
                    "delta_H_peak": float(curr.get("H_peak", np.nan) - prev.get("H_peak", np.nan)),
                    "delta_E_peak": float(curr.get("E_peak", np.nan) - prev.get("E_peak", np.nan)),
                    "delta_Emb_peak": float(curr.get("Emb_peak", np.nan) - prev.get("Emb_peak", np.nan)),
                    "delta_drop_advantage": float(curr.get("drop_advantage", np.nan) - prev.get("drop_advantage", np.nan)),
                    "delta_keep_advantage": float(curr.get("keep_advantage", np.nan) - prev.get("keep_advantage", np.nan)),
                    "delta_pca_peak_mass_pc1": float(curr.get("pca_peak_mass_pc1", np.nan) - prev.get("pca_peak_mass_pc1", np.nan)),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_onset_table(feature_df: pd.DataFrame, sustain_frac: float = 0.8, q: float = 0.75, alignment_mode: str = "step") -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    use_progress = (
        str(alignment_mode).strip().lower() == "progress"
        and "checkpoint_progress_pct" in feature_df.columns
        and pd.to_numeric(feature_df["checkpoint_progress_pct"], errors="coerce").notna().any()
    )
    coord_col = "checkpoint_progress_pct" if use_progress else "checkpoint"
    cps = sorted(pd.to_numeric(feature_df[coord_col], errors="coerce").dropna().astype(int).unique().tolist())
    if not cps:
        return pd.DataFrame()
    c0 = cps[0]
    ref = feature_df[pd.to_numeric(feature_df[coord_col], errors="coerce") == c0]["H_peak"].dropna().to_numpy(dtype=np.float64)
    if ref.size == 0:
        threshold = float(feature_df["H_peak"].quantile(q))
    else:
        threshold = float(np.quantile(ref, q))

    rows: List[dict] = []
    keys = ["track", "variant_stage", "variant_index", "seed"]
    if "B" in feature_df.columns:
        keys.append("B")
    keys.extend(["probe_band", "probe_type"])
    for key, g in feature_df.groupby(keys):
        g = g.sort_values(coord_col)
        vals = g["H_peak"].to_numpy(dtype=np.float64)
        cks = pd.to_numeric(g[coord_col], errors="coerce").to_numpy(dtype=np.int64)
        drop_adv_series = g["drop_advantage"] if "drop_advantage" in g.columns else pd.Series(np.nan, index=g.index)
        keep_adv_series = g["keep_advantage"] if "keep_advantage" in g.columns else pd.Series(np.nan, index=g.index)
        drop_adv = pd.to_numeric(drop_adv_series, errors="coerce").to_numpy(dtype=np.float64)
        keep_adv = pd.to_numeric(keep_adv_series, errors="coerce").to_numpy(dtype=np.float64)
        onset = np.nan
        for i in range(len(vals)):
            if not np.isfinite(vals[i]) or vals[i] < threshold:
                continue
            tail = vals[i:]
            ok = np.isfinite(tail)
            if ok.sum() == 0:
                continue
            frac = float((tail[ok] >= threshold).mean())
            if frac >= float(sustain_frac):
                onset = float(cks[i])
                break
        rows.append(
            {
                "track": key[0],
                "variant_stage": key[1],
                "variant_index": int(key[2]),
                "seed": int(key[3]),
                "B": int(key[4]) if "B" in feature_df.columns else np.nan,
                "probe_band": key[5] if "B" in feature_df.columns else key[4],
                "probe_type": key[6] if "B" in feature_df.columns else key[5],
                "H_peak_onset_checkpoint": onset,
                "H_peak_threshold": threshold,
                "drop_advantage_onset_checkpoint": _sustained_positive_onset(cks, drop_adv, sustain_frac=sustain_frac),
                "keep_advantage_onset_checkpoint": _sustained_positive_onset(cks, keep_adv, sustain_frac=sustain_frac),
                "H_peak_early_slope": _early_slope(cks.astype(np.float64), vals),
                "drop_advantage_early_slope": _early_slope(cks.astype(np.float64), drop_adv),
                "keep_advantage_early_slope": _early_slope(cks.astype(np.float64), keep_adv),
            }
        )
    return pd.DataFrame(rows)


def classify_transitions(map_df: pd.DataFrame, taxonomy_checkpoint: int) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    checkpoint_series = pd.to_numeric(map_df.get("checkpoint", np.nan), errors="coerce")
    has_finite_checkpoint = bool(np.isfinite(checkpoint_series.to_numpy(dtype=float)).any())

    if has_finite_checkpoint:
        ref = map_df[map_df["checkpoint"] == int(taxonomy_checkpoint)].copy()
        if ref.empty:
            cp_max = int(np.nanmax(checkpoint_series.to_numpy(dtype=float)))
            ref = map_df[map_df["checkpoint"] == cp_max].copy()
    else:
        ref = map_df.copy()

    traj_cols = _trajectory_cols(map_df)
    agg = (
        ref.groupby([*traj_cols, "transition", "variant_stage", "prev_variant_stage"], as_index=False)
        .agg(
            tok_gain=("tok_gain", "mean"),
            thr_gain=("thr_gain", "mean"),
            d_H_peak=("delta_H_peak", "mean"),
            d_drop_adv=("delta_drop_advantage", "mean"),
            d_pca=("delta_pca_peak_mass_pc1", "mean"),
            d_grad_rankme=("d_grad_rankme_final", "mean"),
            d_grad_alpha_head=("d_grad_alpha_head_final", "mean"),
            d_grad_alpha_tail=("d_grad_alpha_tail_final", "mean"),
        )
        .copy()
    )

    feature_terms = [
        zscore_series(agg["d_H_peak"]).fillna(0.0),
        zscore_series(agg["d_drop_adv"]).fillna(0.0),
        zscore_series(agg["d_pca"]).fillna(0.0),
    ]
    agg["feature_signal"] = (feature_terms[0] + feature_terms[1] + feature_terms[2]) / 3.0
    agg["optimization_signal"] = (
        zscore_series(agg["d_grad_rankme"].abs()).fillna(0.0)
        + zscore_series(agg["d_grad_alpha_head"].abs()).fillna(0.0)
        + zscore_series(agg["d_grad_alpha_tail"].abs()).fillna(0.0)
    ) / 3.0

    feat_thr = float(agg["feature_signal"].quantile(0.6)) if len(agg) > 0 else 0.0
    opt_thr = float(agg["optimization_signal"].quantile(0.6)) if len(agg) > 0 else 0.0

    labels: List[str] = []
    for _, row in agg.iterrows():
        tok = float(row["tok_gain"])
        thr = float(row["thr_gain"])
        feat = float(row["feature_signal"])
        opt = float(row["optimization_signal"])
        if tok > 0.02 and feat >= feat_thr and abs(thr) <= 0.05:
            labels.append("feature-dominant learning gains")
        elif tok > 0.02 and opt >= opt_thr and feat < feat_thr:
            labels.append("optimization-dominant gains")
        elif tok <= 0.0 and thr > 0.02:
            labels.append("execution-only gains")
        else:
            labels.append("near-null transitions")
    agg["variant_taxonomy"] = labels
    agg["taxonomy_checkpoint"] = int(taxonomy_checkpoint) if has_finite_checkpoint else np.nan
    return agg


def compute_correlations(map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    predictors = ["delta_H_peak", "delta_drop_advantage", "delta_pca_peak_mass_pc1"]
    responses = [
        "tok_gain",
        "thr_gain",
        "constant_loss_tok_gain",
        "constant_loss_time_gain",
        "loss_proximal_tok_gain_mean",
        "loss_proximal_time_gain_mean",
        "loss_proximal_js_div_mean",
        "d_alpha_head_final",
        "d_grad_rankme_final",
    ]
    group_cols = ["checkpoint", "probe_band"] + (["probe_type"] if "probe_type" in map_df.columns else [])
    for group_key, g in map_df.groupby(group_cols):
        if "probe_type" in map_df.columns:
            checkpoint, probe_band, probe_type = group_key
        else:
            checkpoint, probe_band = group_key
            probe_type = ""
        for px in predictors:
            if px not in g.columns:
                continue
            for ry in responses:
                if ry not in g.columns:
                    continue
                x = pd.to_numeric(g[px], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(g[ry], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                rho = safe_spearman(x[mask], y[mask]) if int(mask.sum()) >= 3 else float("nan")
                rows.append(
                    {
                        "checkpoint": int(checkpoint),
                        "probe_band": probe_band,
                        "probe_type": probe_type,
                        "predictor": px,
                        "response": ry,
                        "spearman_r": rho,
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def compute_early_prediction_table(map_df: pd.DataFrame, early_checkpoints: Sequence[int]) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    traj_cols = _trajectory_cols(map_df)
    response_metrics = [
        "loss_proximal_time_gain_mean",
        "loss_proximal_tok_gain_mean",
        "constant_loss_time_gain",
        "constant_loss_tok_gain",
        "tok_gain",
    ]

    rows: List[dict] = []
    for response_metric in response_metrics:
        if response_metric not in map_df.columns:
            continue
        if not np.isfinite(pd.to_numeric(map_df[response_metric], errors="coerce").to_numpy(dtype=float)).any():
            continue
        for ck in early_checkpoints:
            sub = map_df[map_df["checkpoint"] == int(ck)]
            if sub.empty:
                rows.append(
                    {
                        "checkpoint": int(ck),
                        "early_predictor": "delta_H_peak",
                        "response_metric": response_metric,
                        "abs_spearman_with_response": np.nan,
                        "abs_spearman_final_activation_endpoint": np.nan,
                        "beats_final_activation_endpoint": False,
                        "n": 0,
                    }
                )
                continue
            agg = (
                sub.groupby([*traj_cols, "transition"], as_index=False)
                .agg(
                    delta_H_peak=("delta_H_peak", "mean"),
                    d_alpha_head_final=("d_alpha_head_final", "mean"),
                    response_value=(response_metric, "mean"),
                )
                .copy()
            )
            early_x = pd.to_numeric(agg["delta_H_peak"], errors="coerce").to_numpy(dtype=float)
            endpoint_x = pd.to_numeric(agg["d_alpha_head_final"], errors="coerce").to_numpy(dtype=float)
            response_y = pd.to_numeric(agg["response_value"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(early_x) & np.isfinite(endpoint_x) & np.isfinite(response_y)
            corr = abs(safe_spearman(early_x[mask], response_y[mask])) if int(mask.sum()) >= 3 else np.nan
            final_corr = abs(safe_spearman(endpoint_x[mask], response_y[mask])) if int(mask.sum()) >= 3 else np.nan
            rows.append(
                {
                    "checkpoint": int(ck),
                    "early_predictor": "delta_H_peak",
                    "response_metric": response_metric,
                    "abs_spearman_with_response": corr,
                    "abs_spearman_final_activation_endpoint": final_corr,
                    "beats_final_activation_endpoint": bool(np.isfinite(corr) and np.isfinite(final_corr) and corr > final_corr),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_transition_stats(
    map_df: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    metrics = ["delta_H_peak", "delta_drop_advantage", "delta_pca_peak_mass_pc1", "delta_keep_advantage"]
    work = map_df.copy()
    for m in metrics:
        if m not in work.columns:
            work[m] = np.nan
    rows: List[dict] = []

    # pooled by checkpoint/probe band
    group_cols = ["checkpoint", "probe_band"] + (["probe_type"] if "probe_type" in work.columns else [])
    for group_key, g in work.groupby(group_cols):
        if "probe_type" in work.columns:
            checkpoint, probe_band, probe_type = group_key
        else:
            checkpoint, probe_band = group_key
            probe_type = ""
        for m in metrics:
            vals = g[m].to_numpy(dtype=np.float64)
            ci_lo, ci_med, ci_hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            p = sign_flip_pvalue(vals, n_perm=n_perm, seed=seed + 13)
            rows.append(
                {
                    "scope": "pooled",
                    "transition": "ALL",
                    "checkpoint": int(checkpoint),
                    "probe_band": probe_band,
                    "probe_type": probe_type,
                    "metric": m,
                    "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                    "ci_lo": ci_lo,
                    "ci_med": ci_med,
                    "ci_hi": ci_hi,
                    "perm_p": p,
                    "n": int(np.isfinite(vals).sum()),
                }
            )

    # edge-level
    edge_group_cols = ["transition", "checkpoint", "probe_band"] + (["probe_type"] if "probe_type" in work.columns else [])
    for group_key, g in work.groupby(edge_group_cols):
        if "probe_type" in work.columns:
            transition, checkpoint, probe_band, probe_type = group_key
        else:
            transition, checkpoint, probe_band = group_key
            probe_type = ""
        for m in metrics:
            vals = g[m].to_numpy(dtype=np.float64)
            ci_lo, ci_med, ci_hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            p = sign_flip_pvalue(vals, n_perm=n_perm, seed=seed + 29)
            rows.append(
                {
                    "scope": "edge",
                    "transition": transition,
                    "checkpoint": int(checkpoint),
                    "probe_band": probe_band,
                    "probe_type": probe_type,
                    "metric": m,
                    "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                    "ci_lo": ci_lo,
                    "ci_med": ci_med,
                    "ci_hi": ci_hi,
                    "perm_p": p,
                    "n": int(np.isfinite(vals).sum()),
                }
            )

    return pd.DataFrame(rows)


def summarize_transition_map(map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()

    work = map_df.copy()
    for col in [
        "delta_H_peak",
        "delta_drop_advantage",
        "delta_keep_advantage",
        "delta_pca_peak_mass_pc1",
        "causal_delta_keep_vs_ctrl",
        "causal_delta_drop_vs_ctrl",
    ]:
        if col not in work.columns:
            work[col] = np.nan

    group_cols = ["transition", "variant_stage", "prev_variant_stage", "checkpoint", "probe_band"] + (["probe_type"] if "probe_type" in work.columns else [])
    summary = (
        work.groupby(group_cols, as_index=False)
        .agg(
            tok_gain_mean=("tok_gain", "mean"),
            time_gain_mean=("time_gain", "mean"),
            thr_gain_mean=("thr_gain", "mean"),
            constant_loss_tok_gain_mean=("constant_loss_tok_gain", "mean"),
            constant_loss_time_gain_mean=("constant_loss_time_gain", "mean"),
            loss_proximal_tok_gain_mean=("loss_proximal_tok_gain_mean", "mean"),
            loss_proximal_time_gain_mean=("loss_proximal_time_gain_mean", "mean"),
            loss_proximal_js_div_mean=("loss_proximal_js_div_mean", "mean"),
            delta_H_peak_mean=("delta_H_peak", "mean"),
            delta_drop_advantage_mean=("delta_drop_advantage", "mean"),
            delta_keep_advantage_mean=("delta_keep_advantage", "mean"),
            delta_pca_peak_mass_pc1_mean=("delta_pca_peak_mass_pc1", "mean"),
            causal_delta_keep_vs_ctrl_mean=("causal_delta_keep_vs_ctrl", "mean"),
            causal_delta_drop_vs_ctrl_mean=("causal_delta_drop_vs_ctrl", "mean"),
            n=("transition", "size"),
        )
        .copy()
    )
    return summary


def build_alignment_checks(ablation_dir: Path, map_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    ck = pd.to_numeric(map_df.get("checkpoint", np.nan), errors="coerce") if not map_df.empty else pd.Series(dtype=float)
    checkpoint_ready = bool(not map_df.empty and np.isfinite(ck.to_numpy(dtype=float)).any())

    rows.append(
        {
            "alignment_mode": "checkpoint_aligned",
            "available": checkpoint_ready,
            "status": "ok" if checkpoint_ready else "insufficient_checkpoint_level_feature_rows",
            "notes": "Transition map computed with matched checkpoints.",
        }
    )

    constant_loss_ready = bool(
        not map_df.empty
        and (
            np.isfinite(pd.to_numeric(map_df.get("constant_loss_time_gain", np.nan), errors="coerce").to_numpy(dtype=float)).any()
            or np.isfinite(pd.to_numeric(map_df.get("constant_loss_tok_gain", np.nan), errors="coerce").to_numpy(dtype=float)).any()
        )
    )
    rows.append(
        {
            "alignment_mode": "constant_loss_target",
            "available": constant_loss_ready,
            "status": "ready" if constant_loss_ready else "no_paired_target_hits",
            "notes": "Constant-loss gains require both sides of a transition to hit the configured target loss.",
        }
    )

    loss_proximal_ready = bool(
        not map_df.empty
        and (
            np.isfinite(pd.to_numeric(map_df.get("loss_proximal_time_gain_mean", np.nan), errors="coerce").to_numpy(dtype=float)).any()
            or np.isfinite(pd.to_numeric(map_df.get("loss_proximal_tok_gain_mean", np.nan), errors="coerce").to_numpy(dtype=float)).any()
        )
    )
    rows.append(
        {
            "alignment_mode": "loss_proximal",
            "available": loss_proximal_ready,
            "status": "ready" if loss_proximal_ready else "insufficient_overlap_or_missing_run_artifacts",
            "notes": "Loss-proximal gains compare predecessor and successor runs at shared loss levels using matched checkpoints.",
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-variant feature-learning transition map and diagnostics.")
    parser.add_argument("--ablation-dir", type=str, default="toy_model/variant_concat_ablation")
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--feature-dir", type=str, default="")
    parser.add_argument("--alignment-mode", type=str, default="step", choices=["step", "progress"])
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--checkpoint-progress-pcts", type=str, default="")
    parser.add_argument("--bands", type=str, default="97:200,179:50")
    parser.add_argument("--early-checkpoints", type=str, default="400,800,1600")
    parser.add_argument("--early-progress-pcts", type=str, default="")
    parser.add_argument("--taxonomy-checkpoint", type=int, default=1600)
    parser.add_argument("--taxonomy-progress-pct", type=int, default=50)
    parser.add_argument("--num-loss-proximal-targets", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (ablation_dir / "variant_concat_ablation_summary.csv")
    feature_dir = Path(args.feature_dir) if args.feature_dir else (ablation_dir / "feature_learning_analysis")
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_int_list(args.checkpoints)
    checkpoint_progress_pcts = parse_int_list(args.checkpoint_progress_pcts)
    bands = parse_band_list(args.bands)
    early_checkpoints = parse_int_list(args.early_checkpoints)
    early_progress_pcts = parse_int_list(args.early_progress_pcts)

    summary_df = pd.read_csv(summary_csv)
    endpoints = build_transition_endpoints(summary_df)
    endpoints = augment_loss_proximal_transition_metrics(
        endpoints,
        ablation_dir=ablation_dir,
        num_targets=int(args.num_loss_proximal_targets),
    )

    fl_summary, fl_causal, fl_pca = load_feature_tables(feature_dir)
    feature_merged = build_feature_merged(fl_summary, fl_causal, fl_pca) if not fl_summary.empty else pd.DataFrame()
    main_probe_type = choose_main_probe_type(fl_summary, fl_causal)

    transition_map = build_transition_feature_map(
        endpoints=endpoints,
        feature_df=feature_merged,
        checkpoints=checkpoint_progress_pcts if args.alignment_mode == "progress" and checkpoint_progress_pcts else checkpoints,
        bands=bands,
        alignment_mode=args.alignment_mode,
    )
    if not transition_map.empty and "probe_type" in transition_map.columns:
        preferred = transition_map[transition_map["probe_type"] == main_probe_type].copy()
        if not preferred.empty:
            transition_map = preferred

    taxonomy_ref = int(args.taxonomy_progress_pct) if args.alignment_mode == "progress" else int(args.taxonomy_checkpoint)
    taxonomy_df = classify_transitions(transition_map, taxonomy_checkpoint=taxonomy_ref)
    if not taxonomy_df.empty and not transition_map.empty:
        traj_cols = _trajectory_cols(transition_map)
        transition_map = transition_map.merge(
            taxonomy_df[[*traj_cols, "transition", "variant_taxonomy", "feature_signal", "optimization_signal"]],
            on=[*traj_cols, "transition"],
            how="left",
        )

    onset_df = (
        compute_onset_table(feature_merged[feature_merged["probe_type"] == main_probe_type], alignment_mode=args.alignment_mode)
        if not feature_merged.empty and "probe_type" in feature_merged.columns
        else (compute_onset_table(feature_merged, alignment_mode=args.alignment_mode) if not feature_merged.empty else pd.DataFrame())
    )
    corr_df = compute_correlations(transition_map)
    early_ref = early_progress_pcts if args.alignment_mode == "progress" and early_progress_pcts else early_checkpoints
    early_df = compute_early_prediction_table(transition_map, early_checkpoints=early_ref)
    transition_summary_df = summarize_transition_map(transition_map)
    stats_df = compute_transition_stats(
        transition_map,
        n_boot=int(args.n_bootstrap),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )
    align_df = build_alignment_checks(ablation_dir, transition_map)

    map_cols = [
        "track",
        "seed",
        "prev_variant_index",
        "variant_index",
        "prev_variant_stage",
        "variant_stage",
        "prev_variant_combo",
        "variant_combo",
        "transition",
        "B",
        "checkpoint",
        "checkpoint_step",
        "alignment_mode",
        "alignment_key",
        "checkpoint_progress_pct",
        "checkpoint_progress_frac",
        "probe_band",
        "probe_type",
        "tok_gain",
        "thr_gain",
        "time_gain",
        "log_tok_gain",
        "log_thr_gain",
        "prev_hit_target",
        "hit_target",
        "both_hit_target",
        "constant_loss_tok_gain",
        "constant_loss_time_gain",
        "log_constant_loss_tok_gain",
        "log_constant_loss_time_gain",
        "loss_proximal_tok_gain_mean",
        "loss_proximal_time_gain_mean",
        "loss_proximal_js_div_mean",
        "loss_proximal_loss_gap_mean",
        "loss_proximal_n_targets",
        "H_peak",
        "E_peak",
        "Emb_peak",
        "keep_key_loss",
        "drop_key_loss",
        "keep_ctrl_loss",
        "drop_ctrl_loss",
        "delta_keep",
        "delta_drop",
        "hidden_delta_keep",
        "hidden_delta_drop",
        "causal_delta_keep",
        "causal_delta_drop",
        "causal_delta_keep_ctrl",
        "causal_delta_drop_ctrl",
        "causal_delta_keep_vs_ctrl",
        "causal_delta_drop_vs_ctrl",
        "causal_metric_source",
        "dominant_freq",
        "pca_peak_mass_pc1",
        "delta_H_peak",
        "delta_E_peak",
        "delta_Emb_peak",
        "delta_drop_advantage",
        "delta_keep_advantage",
        "delta_pca_peak_mass_pc1",
        "d_rankme_final",
        "d_alpha_head_final",
        "d_alpha_tail_final",
        "d_grad_rankme_final",
        "d_grad_alpha_head_final",
        "d_grad_alpha_tail_final",
        "variant_taxonomy",
        "feature_signal",
        "optimization_signal",
    ]
    if transition_map.empty:
        transition_map = pd.DataFrame(columns=map_cols)
    else:
        extra = [c for c in transition_map.columns if c not in map_cols]
        transition_map = transition_map.reindex(columns=map_cols + extra)

    onset_cols = [
        "track",
        "variant_stage",
        "variant_index",
        "seed",
        "B",
        "probe_band",
        "probe_type",
        "H_peak_onset_checkpoint",
        "H_peak_threshold",
        "drop_advantage_onset_checkpoint",
        "keep_advantage_onset_checkpoint",
        "H_peak_early_slope",
        "drop_advantage_early_slope",
        "keep_advantage_early_slope",
    ]
    if onset_df.empty:
        onset_df = pd.DataFrame(columns=onset_cols)
    else:
        onset_df = onset_df.reindex(columns=onset_cols)

    taxonomy_cols = [
        "track",
        "seed",
        "B",
        "transition",
        "variant_stage",
        "prev_variant_stage",
        "tok_gain",
        "thr_gain",
        "d_H_peak",
        "d_drop_adv",
        "d_pca",
        "d_grad_rankme",
        "d_grad_alpha_head",
        "d_grad_alpha_tail",
        "feature_signal",
        "optimization_signal",
        "variant_taxonomy",
        "taxonomy_checkpoint",
    ]
    if taxonomy_df.empty:
        taxonomy_df = pd.DataFrame(columns=taxonomy_cols)
    else:
        taxonomy_df = taxonomy_df.reindex(columns=taxonomy_cols)

    corr_cols = ["checkpoint", "probe_band", "probe_type", "predictor", "response", "spearman_r", "n"]
    if corr_df.empty:
        corr_df = pd.DataFrame(columns=corr_cols)
    else:
        corr_df = corr_df.reindex(columns=corr_cols)

    early_cols = [
        "checkpoint",
        "early_predictor",
        "response_metric",
        "abs_spearman_with_response",
        "abs_spearman_final_activation_endpoint",
        "beats_final_activation_endpoint",
        "n",
    ]
    if early_df.empty:
        early_df = pd.DataFrame(columns=early_cols)
    else:
        early_df = early_df.reindex(columns=early_cols)

    transition_summary_cols = [
        "transition",
        "variant_stage",
        "prev_variant_stage",
        "checkpoint",
        "probe_band",
        "probe_type",
        "tok_gain_mean",
        "time_gain_mean",
        "thr_gain_mean",
        "constant_loss_tok_gain_mean",
        "constant_loss_time_gain_mean",
        "loss_proximal_tok_gain_mean",
        "loss_proximal_time_gain_mean",
        "loss_proximal_js_div_mean",
        "delta_H_peak_mean",
        "delta_drop_advantage_mean",
        "delta_keep_advantage_mean",
        "delta_pca_peak_mass_pc1_mean",
        "causal_delta_keep_vs_ctrl_mean",
        "causal_delta_drop_vs_ctrl_mean",
        "n",
    ]
    if transition_summary_df.empty:
        transition_summary_df = pd.DataFrame(columns=transition_summary_cols)
    else:
        transition_summary_df = transition_summary_df.reindex(columns=transition_summary_cols)

    stats_cols = ["scope", "transition", "checkpoint", "probe_band", "probe_type", "metric", "mean", "ci_lo", "ci_med", "ci_hi", "perm_p", "n"]
    if stats_df.empty:
        stats_df = pd.DataFrame(columns=stats_cols)
    else:
        stats_df = stats_df.reindex(columns=stats_cols)

    align_cols = ["alignment_mode", "available", "status", "notes"]
    if align_df.empty:
        align_df = pd.DataFrame(columns=align_cols)
    else:
        align_df = align_df.reindex(columns=align_cols)

    # Standardized required artifact.
    transition_map.to_csv(out_dir / "feature_variant_transition_map.csv", index=False)

    # Companion diagnostics.
    onset_df.to_csv(out_dir / "feature_variant_onset.csv", index=False)
    transition_summary_df.to_csv(out_dir / "feature_transition_summary.csv", index=False)
    taxonomy_df.to_csv(out_dir / "feature_variant_taxonomy.csv", index=False)
    corr_df.to_csv(out_dir / "feature_transition_correlations.csv", index=False)
    early_df.to_csv(out_dir / "feature_early_prediction.csv", index=False)
    stats_df.to_csv(out_dir / "feature_transition_stats.csv", index=False)
    align_df.to_csv(out_dir / "feature_alignment_checks.csv", index=False)

    summary_note = pd.DataFrame(
        [
            {
                "main_probe_type": main_probe_type,
                "main_causal_signal": "hidden_state_if_available_else_embedding",
                "alignment_mode": str(args.alignment_mode),
                "notes": "Transition map prefers matched_band when present and falls back to clean_band; causal summaries prefer hidden-state projections when available.",
            }
        ]
    )
    summary_note.to_csv(out_dir / "feature_variant_map_protocol.csv", index=False)

    print(f"Wrote: {out_dir / 'feature_variant_transition_map.csv'}")
    print(f"Wrote: {out_dir / 'feature_variant_onset.csv'}")
    print(f"Wrote: {out_dir / 'feature_transition_summary.csv'}")
    print(f"Wrote: {out_dir / 'feature_variant_taxonomy.csv'}")
    print(f"Wrote: {out_dir / 'feature_transition_correlations.csv'}")
    print(f"Wrote: {out_dir / 'feature_early_prediction.csv'}")
    print(f"Wrote: {out_dir / 'feature_transition_stats.csv'}")
    print(f"Wrote: {out_dir / 'feature_alignment_checks.csv'}")
    print(f"Wrote: {out_dir / 'feature_variant_map_protocol.csv'}")


if __name__ == "__main__":
    main()
