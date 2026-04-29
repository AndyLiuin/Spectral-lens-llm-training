from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _text_only(ax, msg: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)


def _choose_main_probe_type(*dfs: pd.DataFrame) -> str:
    for df in dfs:
        if df.empty or "probe_type" not in df.columns:
            continue
        vals = {str(x).strip() for x in df["probe_type"].dropna().unique()}
        if "matched_band" in vals:
            return "matched_band"
        if "clean_band" in vals:
            return "clean_band"
    return "clean_band"


def _filter_probe_type(df: pd.DataFrame, probe_type: str) -> pd.DataFrame:
    if df.empty or "probe_type" not in df.columns:
        return df.copy()
    sub = df[df["probe_type"] == probe_type].copy()
    return sub if not sub.empty else df.copy()


def _preferred_causal_advantages(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    hidden_cols = {"hidden_delta_keep", "hidden_delta_drop", "hidden_delta_keep_ctrl", "hidden_delta_drop_ctrl"}
    if hidden_cols.issubset(df.columns):
        keep = pd.to_numeric(df["hidden_delta_keep"], errors="coerce") - pd.to_numeric(df["hidden_delta_keep_ctrl"], errors="coerce")
        drop = pd.to_numeric(df["hidden_delta_drop_ctrl"], errors="coerce") - pd.to_numeric(df["hidden_delta_drop"], errors="coerce")
        if np.isfinite(keep.to_numpy(dtype=float)).any() or np.isfinite(drop.to_numpy(dtype=float)).any():
            return keep, drop, "hidden-state"
    keep = pd.to_numeric(df.get("delta_keep", np.nan), errors="coerce") - pd.to_numeric(df.get("delta_keep_ctrl", np.nan), errors="coerce")
    drop = pd.to_numeric(df.get("delta_drop_ctrl", np.nan), errors="coerce") - pd.to_numeric(df.get("delta_drop", np.nan), errors="coerce")
    return keep, drop, "embedding"


def _preferred_transition_response(df: pd.DataFrame) -> tuple[str, str]:
    for col, label in [
        ("loss_proximal_time_gain_mean", "Loss-Proximal Time Gain"),
        ("loss_proximal_tok_gain_mean", "Loss-Proximal Token Gain"),
        ("constant_loss_time_gain", "Constant-Loss Time Gain"),
        ("constant_loss_tok_gain", "Constant-Loss Token Gain"),
        ("tok_gain", "TokGain"),
    ]:
        if col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)).any():
            return col, label
    return "tok_gain", "TokGain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot feature-learning toy 2x2 panel + appendix diagnostics.")
    parser.add_argument("--feature-dir", type=str, default="toy_model/variant_concat_ablation/feature_learning_analysis")
    parser.add_argument("--out-main", type=str, default="")
    parser.add_argument("--out-appendix-seeds", type=str, default="")
    parser.add_argument("--out-appendix-bands", type=str, default="")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_main = Path(args.out_main) if args.out_main else (feature_dir / "feature_learning_main_2x2.png")
    out_seed = Path(args.out_appendix_seeds) if args.out_appendix_seeds else (feature_dir / "feature_learning_appendix_seed_overlays.png")
    out_band = Path(args.out_appendix_bands) if args.out_appendix_bands else (feature_dir / "feature_learning_appendix_band_robustness.png")
    out_main.parent.mkdir(parents=True, exist_ok=True)

    summary = _read_csv(feature_dir / "feature_learning_summary.csv")
    causal = _read_csv(feature_dir / "feature_causal_effects.csv")
    pca = _read_csv(feature_dir / "feature_pca_summary.csv")
    tmap = _read_csv(feature_dir / "feature_variant_transition_map.csv")
    main_probe_type = _choose_main_probe_type(summary, causal, pca, tmap)
    summary_main = _filter_probe_type(summary, main_probe_type)
    causal_main = _filter_probe_type(causal, main_probe_type)
    pca_main = _filter_probe_type(pca, main_probe_type)
    tmap_main = _filter_probe_type(tmap, main_probe_type)

    # Main 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: H[m] evolution (uniform -> spiky proxy).
    ax = axes[0, 0]
    if not summary_main.empty and {"checkpoint", "variant_stage", "H_peak"}.issubset(summary_main.columns):
        g = (
            summary_main.groupby(["variant_stage", "checkpoint"], as_index=False)["H_peak"]
            .mean()
            .sort_values(["variant_stage", "checkpoint"])
        )
        variants = list(g["variant_stage"].dropna().unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(variants)))
        for color, v in zip(cmap, variants):
            gv = g[g["variant_stage"] == v]
            ax.plot(gv["checkpoint"], gv["H_peak"], marker="o", ms=3, lw=1.5, color=color, label=v)
        ax.set_title(f"H[m] Fourier Concentration Over Training ({main_probe_type})")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("H_peak")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    else:
        _text_only(ax, "No checkpoint feature summary available yet.\nRun probe pipeline after checkpoint reruns.")

    # Top-right: causal keep/drop vs controls.
    ax = axes[0, 1]
    if not causal_main.empty and {"variant_stage", "checkpoint"}.issubset(causal_main.columns):
        final_ck = int(causal_main["checkpoint"].dropna().max())
        c = causal_main[causal_main["checkpoint"] == final_ck].copy()
        keep_adv, drop_adv, causal_scope = _preferred_causal_advantages(c)
        c["keep_advantage"] = keep_adv
        c["drop_advantage"] = drop_adv
        agg = c.groupby("variant_stage", as_index=False)[["keep_advantage", "drop_advantage"]].mean()

        x = np.arange(len(agg))
        w = 0.38
        ax.bar(x - w / 2, agg["keep_advantage"], width=w, label="keep key vs ctrl")
        ax.bar(x + w / 2, agg["drop_advantage"], width=w, label="drop key vs ctrl")
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["variant_stage"], rotation=30, ha="right")
        ax.set_title(f"{causal_scope.title()} Causal Effect Sizes @ ckpt={final_ck}")
        ax.set_ylabel("Effect size")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    else:
        _text_only(ax, "No causal checkpoint table found.")

    # Bottom-left: PCA dominant frequency heatmap.
    ax = axes[1, 0]
    if not pca_main.empty and {"pc_index", "variant_stage", "checkpoint", "dominant_freq"}.issubset(pca_main.columns):
        p = pca_main[pca_main["pc_index"] == 1].copy()
        if not p.empty:
            heat = p.groupby(["variant_stage", "checkpoint"], as_index=False)["dominant_freq"].mean()
            piv = heat.pivot(index="variant_stage", columns="checkpoint", values="dominant_freq").sort_index()
            im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="magma")
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels(piv.index)
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels([str(int(c)) for c in piv.columns], rotation=45, ha="right")
            ax.set_title(f"PC1 Dominant Frequency ({main_probe_type})")
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Variant")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Dominant frequency")
        else:
            _text_only(ax, "PCA table is empty.")
    else:
        _text_only(ax, "No PCA summary table found.")

    # Bottom-right: feature vs spectral/token alignment.
    ax = axes[1, 1]
    response_col, response_label = _preferred_transition_response(tmap_main)
    color_col = "loss_proximal_js_div_mean" if "loss_proximal_js_div_mean" in tmap_main.columns else "thr_gain"
    group_keys = [c for c in ["track", "seed", "B", "transition"] if c in tmap_main.columns]
    if not tmap_main.empty and {"delta_H_peak", response_col}.issubset(tmap_main.columns):
        m = (
            tmap_main.groupby(group_keys, as_index=False)
            .agg(delta_H_peak=("delta_H_peak", "mean"), response_value=(response_col, "mean"), color_value=(color_col, "mean"))
            .dropna(subset=["delta_H_peak", "response_value"])
        )
        if not m.empty:
            sc = ax.scatter(m["delta_H_peak"], m["response_value"], c=m["color_value"], cmap="coolwarm", s=80, edgecolors="white", linewidths=0.7)
            ax.axhline(0.0, color="gray", lw=1)
            ax.axvline(0.0, color="gray", lw=1)
            ax.set_title(f"Feature Delta vs {response_label} ({main_probe_type})")
            ax.set_xlabel("delta_H_peak")
            ax.set_ylabel(response_label)
            ax.grid(True, alpha=0.3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Loss-Proximal JS" if color_col == "loss_proximal_js_div_mean" else "ThrGain")
        else:
            _text_only(ax, "Transition map found but no finite alignment rows.")
    else:
        _text_only(ax, "No transition map with feature deltas found.")

    fig.tight_layout()
    fig.savefig(out_main, dpi=220)
    plt.close(fig)

    # Appendix: seed overlays
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    if not summary_main.empty and {"checkpoint", "variant_stage", "seed", "H_peak"}.issubset(summary_main.columns):
        for (variant, seed), g in summary_main.groupby(["variant_stage", "seed"]):
            g = g.sort_values("checkpoint")
            ax2.plot(g["checkpoint"], g["H_peak"], alpha=0.45, lw=1.2)
        grand = summary_main.groupby("checkpoint", as_index=False)["H_peak"].mean().sort_values("checkpoint")
        ax2.plot(grand["checkpoint"], grand["H_peak"], color="black", lw=2.5, label="mean")
        ax2.set_title(f"Appendix: H_peak Seed Overlays ({main_probe_type})")
        ax2.set_xlabel("Checkpoint")
        ax2.set_ylabel("H_peak")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        _text_only(ax2, "No summary rows for seed overlays.")
    fig2.tight_layout()
    fig2.savefig(out_seed, dpi=220)
    plt.close(fig2)

    # Appendix: band robustness
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if not summary.empty and {"checkpoint", "probe_band", "H_peak"}.issubset(summary.columns):
        ck = int(summary["checkpoint"].dropna().max())
        b = summary[summary["checkpoint"] == ck].copy()
        if "probe_type" in b.columns:
            b["probe_label"] = b["probe_band"].astype(str) + " / " + b["probe_type"].astype(str)
            agg = b.groupby("probe_label", as_index=False)["H_peak"].mean()
            xvals = agg["probe_label"]
        else:
            agg = b.groupby("probe_band", as_index=False)["H_peak"].mean()
            xvals = agg["probe_band"]
        ax3.bar(xvals, agg["H_peak"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"][: len(agg)])
        ax3.set_title(f"Appendix: Probe Robustness @ ckpt={ck}")
        ax3.set_xlabel("Probe band / regime")
        ax3.set_ylabel("Mean H_peak")
        ax3.grid(True, axis="y", alpha=0.3)
        for tick in ax3.get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")
    else:
        _text_only(ax3, "No summary rows for band robustness.")
    fig3.tight_layout()
    fig3.savefig(out_band, dpi=220)
    plt.close(fig3)

    print(f"Wrote: {out_main}")
    print(f"Wrote: {out_seed}")
    print(f"Wrote: {out_band}")


if __name__ == "__main__":
    main()
