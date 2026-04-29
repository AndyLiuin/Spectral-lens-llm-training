from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


RUN_NAME_RE = re.compile(r"__B-(\d+)__seed-(\d+)__")


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


def parse_run_metadata(run_name: str) -> Dict[str, float]:
    match = RUN_NAME_RE.search(str(run_name))
    if not match:
        return {"batch_size": float("nan"), "actual_seed": float("nan")}
    return {
        "batch_size": float(match.group(1)),
        "actual_seed": float(match.group(2)),
    }


def add_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if "run_name" not in df.columns:
        out = df.copy()
        out["batch_size"] = np.nan
        out["actual_seed"] = np.nan
        return out
    meta = df["run_name"].map(parse_run_metadata).apply(pd.Series)
    out = df.copy()
    out["batch_size"] = pd.to_numeric(meta["batch_size"], errors="coerce")
    out["actual_seed"] = pd.to_numeric(meta["actual_seed"], errors="coerce")
    return out


def build_stage_coverage(summary_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = (
        feature_df.groupby(["variant_stage", "batch_size"], as_index=False)["checkpoint"]
        .agg(lambda s: ",".join(str(int(x)) for x in sorted(pd.Series(s).dropna().astype(int).unique())))
        .rename(columns={"checkpoint": "available_checkpoints"})
    )

    out = (
        summary_df.groupby(["variant_stage", "batch_size"], as_index=False)
        .agg(
            actual_seed_n=("actual_seed", lambda s: int(pd.Series(s).dropna().nunique())),
            actual_seeds=("actual_seed", lambda s: ",".join(str(int(x)) for x in sorted(pd.Series(s).dropna().astype(int).unique()))),
            test_loss=("test_loss", "mean"),
            tokens_seen=("tokens_seen", "mean"),
            train_time_s=("train_time_s", "mean"),
        )
        .merge(checkpoints, on=["variant_stage", "batch_size"], how="left")
        .sort_values(["variant_stage", "batch_size"])
    )
    return out


def build_transition_coverage(map_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["transition", "checkpoint"]
    if "probe_type" in map_df.columns:
        group_cols.append("probe_type")
    if "probe_band" in map_df.columns:
        group_cols.append("probe_band")

    for keys, group in map_df.groupby(group_cols):
        if isinstance(keys, tuple):
            transition, checkpoint, *rest = keys
        else:
            transition, checkpoint, rest = keys, np.nan, []
        probe_type = rest[0] if len(rest) >= 1 else ""
        probe_band = rest[1] if len(rest) >= 2 else ""
        rows.append(
            {
                "transition": transition,
                "checkpoint": int(checkpoint),
                "probe_type": probe_type,
                "probe_band": probe_band,
                "n_rows": int(len(group)),
                "batch_sizes": ",".join(str(int(x)) for x in sorted(group["seed"].dropna().astype(int).unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["transition", "checkpoint", "probe_type", "probe_band"])


def build_early_audit(map_df: pd.DataFrame) -> pd.DataFrame:
    matched = map_df[map_df.get("probe_type", "") == "matched_band"].copy()
    rows: List[dict] = []
    checkpoints = sorted(matched["checkpoint"].dropna().astype(int).unique())

    for checkpoint in checkpoints:
        sub = matched[matched["checkpoint"] == checkpoint].copy()
        agg = (
            sub.groupby(["track", "seed", "transition"], as_index=False)
            .agg(
                delta_H_peak=("delta_H_peak", "mean"),
                tok_gain=("tok_gain", "mean"),
                d_alpha_head_final=("d_alpha_head_final", "mean"),
            )
            .sort_values(["seed", "transition"])
        )
        rows.append(
            {
                "checkpoint": int(checkpoint),
                "n_points": int(len(agg)),
                "batch_sizes": ",".join(str(int(x)) for x in sorted(agg["seed"].dropna().astype(int).unique())),
                "abs_spearman_delta_H_peak_vs_tok_gain": abs(safe_spearman(agg["delta_H_peak"], agg["tok_gain"])),
                "abs_spearman_final_d_alpha_head_vs_tok_gain": abs(
                    safe_spearman(agg["d_alpha_head_final"], agg["tok_gain"])
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("checkpoint")
    if not out.empty:
        out["beats_final_activation_endpoint"] = (
            out["abs_spearman_delta_H_peak_vs_tok_gain"] > out["abs_spearman_final_d_alpha_head_vs_tok_gain"]
        )
    return out


def build_edge_table(map_df: pd.DataFrame, checkpoints: Sequence[int]) -> pd.DataFrame:
    matched = map_df[map_df.get("probe_type", "") == "matched_band"].copy()
    matched = matched[matched["checkpoint"].isin([int(x) for x in checkpoints])].copy()
    if matched.empty:
        return matched
    out = (
        matched.groupby(["checkpoint", "seed", "transition"], as_index=False)
        .agg(
            tok_gain=("tok_gain", "mean"),
            thr_gain=("thr_gain", "mean"),
            delta_H_peak=("delta_H_peak", "mean"),
            d_alpha_head_final=("d_alpha_head_final", "mean"),
        )
        .rename(columns={"seed": "batch_size"})
        .sort_values(["checkpoint", "transition", "batch_size"])
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit toy feature-bridge coverage with claim-safe metadata.")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="toy_model/toy_model_runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="toy_model/toy_model_runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/variant_concat_ablation_summary.csv",
    )
    parser.add_argument("--edge-checkpoints", type=str, default="500,1000,1500")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    summary_csv = Path(args.summary_csv)
    edge_checkpoints = [int(x) for x in str(args.edge_checkpoints).split(",") if x.strip()]

    summary_df = add_run_metadata(pd.read_csv(summary_csv))
    feature_df = add_run_metadata(pd.read_csv(feature_dir / "feature_learning_summary.csv"))
    map_df = pd.read_csv(feature_dir / "feature_variant_transition_map.csv")

    stage_coverage = build_stage_coverage(summary_df=summary_df, feature_df=feature_df)
    transition_coverage = build_transition_coverage(map_df=map_df)
    early_audit = build_early_audit(map_df=map_df)
    edge_table = build_edge_table(map_df=map_df, checkpoints=edge_checkpoints)

    stage_coverage.to_csv(feature_dir / "feature_claimsafe_stage_coverage.csv", index=False)
    transition_coverage.to_csv(feature_dir / "feature_claimsafe_transition_coverage.csv", index=False)
    early_audit.to_csv(feature_dir / "feature_claimsafe_early_audit.csv", index=False)
    edge_table.to_csv(feature_dir / "feature_claimsafe_edge_table.csv", index=False)

    seed_matches_batch = False
    if "seed" in summary_df.columns:
        seed_matches_batch = bool(
            (
                pd.to_numeric(summary_df["seed"], errors="coerce")
                == pd.to_numeric(summary_df["batch_size"], errors="coerce")
            )
            .fillna(False)
            .all()
        )

    best_row = early_audit.sort_values("abs_spearman_delta_H_peak_vs_tok_gain", ascending=False).iloc[0]
    summary = {
        "actual_seeds": sorted(int(x) for x in summary_df["actual_seed"].dropna().astype(int).unique()),
        "batch_sizes": sorted(int(x) for x in summary_df["batch_size"].dropna().astype(int).unique()),
        "summary_seed_column_matches_batch_size": seed_matches_batch,
        "stage_count": int(summary_df["variant_stage"].nunique()),
        "transition_count": int(map_df["transition"].nunique()),
        "best_early_checkpoint": int(best_row["checkpoint"]),
        "best_early_abs_spearman": float(best_row["abs_spearman_delta_H_peak_vs_tok_gain"]),
        "best_final_abs_spearman": float(best_row["abs_spearman_final_d_alpha_head_vs_tok_gain"]),
        "best_checkpoint_n_points": int(best_row["n_points"]),
        "caveat": "Current checkpoint bridge covers one actual seed across multiple batch regimes, not a true multi-seed sweep.",
    }
    (feature_dir / "feature_claimsafe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
