from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE4 = ["baseline", "rope", "muon", "untie_embed"]
TRANSITION_COLORS = {
    "baseline->rope": "#4C78A8",
    "rope->muon": "#F58518",
    "muon->untie_embed": "#54A24B",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_hpeak(summary: pd.DataFrame, out_path: Path, probe_band: str, probe_type: str) -> None:
    sub = summary[(summary["probe_type"] == probe_type) & (summary["probe_band"] == probe_band)].copy()
    sub = sub[sub["variant_stage"].isin(CORE4)]
    sub["variant_stage"] = pd.Categorical(sub["variant_stage"], categories=CORE4, ordered=True)
    agg = (
        sub.groupby(["variant_stage", "checkpoint"], as_index=False)["H_peak"]
        .mean()
        .sort_values(["variant_stage", "checkpoint"])
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(CORE4)))
    for color, variant in zip(cmap, CORE4):
        g = agg[agg["variant_stage"] == variant]
        if g.empty:
            continue
        ax.plot(g["checkpoint"], g["H_peak"], marker="o", ms=4, lw=2, color=color, label=variant)

    ax.set_title(r"Candidate C: $H_{\mathrm{peak}}$ trajectories in the core-four chain")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(r"$H_{\mathrm{peak}}$")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_prediction_correlations(corr: pd.DataFrame, out_path: Path) -> None:
    use = corr[corr["response"] == "tok_gain"].copy()
    use = use[use["predictor"].isin(["delta_H_peak", "delta_pca_peak_mass_pc1"])]
    use["abs_spearman"] = use["spearman_r"].abs()

    bands = [b for b in ["97:200", "179:50"] if b in set(use["probe_band"].dropna())]
    if not bands:
        bands = sorted(use["probe_band"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, len(bands), figsize=(7.2 * max(len(bands), 1), 4.6), squeeze=False)
    color_map = {
        "delta_H_peak": "#4C78A8",
        "delta_pca_peak_mass_pc1": "#E45756",
    }
    label_map = {
        "delta_H_peak": r"$|\rho(\Delta H_{\mathrm{peak}}, \mathrm{tok\ gain})|$",
        "delta_pca_peak_mass_pc1": r"$|\rho(\Delta \mathrm{PC1\ peak}, \mathrm{tok\ gain})|$",
    }

    for ax, band in zip(axes[0], bands):
        band_df = use[(use["probe_type"] == "matched_band") & (use["probe_band"] == band)].copy()
        for predictor in ["delta_H_peak", "delta_pca_peak_mass_pc1"]:
            g = band_df[band_df["predictor"] == predictor].sort_values("checkpoint")
            if g.empty:
                continue
            ax.plot(
                g["checkpoint"],
                g["abs_spearman"],
                marker="o",
                ms=4,
                lw=2,
                color=color_map[predictor],
                label=label_map[predictor],
            )
            for _, row in g.iterrows():
                ax.text(
                    row["checkpoint"],
                    row["abs_spearman"] + 0.02,
                    f"n={int(row['n'])}",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color=color_map[predictor],
                )
        ax.set_title(f"Candidate D1: early-prediction support ({band})")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(r"Absolute Spearman")
        ax.set_ylim(0.0, 1.05)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_ck500_scatter(tmap: pd.DataFrame, out_path: Path) -> None:
    use = tmap[(tmap["checkpoint"] == 500) & (tmap["probe_type"] == "matched_band")].copy()
    use = use[use["transition"].isin(TRANSITION_COLORS)]
    if "B" not in use.columns:
        use["B"] = pd.to_numeric(use["seed"], errors="coerce")

    bands = [b for b in ["97:200", "179:50"] if b in set(use["probe_band"].dropna())]
    if not bands:
        bands = sorted(use["probe_band"].dropna().unique().tolist())

    marker_map = {32: "o", 128: "s", 512: "^"}
    fig, axes = plt.subplots(1, len(bands), figsize=(7.2 * max(len(bands), 1), 4.8), squeeze=False)
    for ax, band in zip(axes[0], bands):
        band_df = use[use["probe_band"] == band].copy()
        agg = (
            band_df.groupby(["transition", "B"], as_index=False)
            .agg(delta_H_peak=("delta_H_peak", "mean"), tok_gain=("tok_gain", "mean"))
            .dropna(subset=["delta_H_peak", "tok_gain"])
        )
        for _, row in agg.iterrows():
            color = TRANSITION_COLORS.get(row["transition"], "#333333")
            marker = marker_map.get(int(row["B"]), "o")
            ax.scatter(
                row["delta_H_peak"],
                row["tok_gain"],
                s=95,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
            )
            ax.text(
                row["delta_H_peak"],
                row["tok_gain"] + 0.015,
                f"{row['transition'].replace('baseline', 'base')}\nB={int(row['B'])}",
                fontsize=7,
                ha="center",
                va="bottom",
            )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"Candidate D2: checkpoint-500 bridge points ({band})")
        ax.set_xlabel(r"$\Delta H_{\mathrm{peak}}$")
        ax.set_ylabel("Token gain")
        _style_axes(ax)

    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate replacement plots for the toy bridge figure.")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=(
            str(
                Path(__file__).resolve().parent
                / "toy_model_runs"
                / "modarith_a_seed0_steps5000_asha_fix_20260331"
                / "concat_batch_regime_selected_runs"
                / "variant_concat_ablation_core4_bybatch"
                / "feature_learning_analysis"
            )
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "paper_figures" / "neurips_figures"),
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)

    summary = _read_csv(feature_dir / "feature_learning_summary.csv")
    corr = _read_csv(feature_dir / "feature_transition_correlations.csv")
    tmap = _read_csv(feature_dir / "feature_variant_transition_map.csv")

    plot_hpeak(
        summary,
        out_dir / "toy_candidate_hpeak_core4.png",
        probe_band="97:200",
        probe_type="matched_band",
    )
    plot_prediction_correlations(corr, out_dir / "toy_candidate_prediction_correlations.png")
    plot_ck500_scatter(tmap, out_dir / "toy_candidate_ck500_bridge_scatter.png")

    print(f"Wrote: {out_dir / 'toy_candidate_hpeak_core4.png'}")
    print(f"Wrote: {out_dir / 'toy_candidate_prediction_correlations.png'}")
    print(f"Wrote: {out_dir / 'toy_candidate_ck500_bridge_scatter.png'}")


if __name__ == "__main__":
    main()
