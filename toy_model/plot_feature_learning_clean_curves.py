from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


DEFAULT_VARIANT_ORDER = ["baseline", "rope", "muon", "untie_embed"]
DEFAULT_COLORS = {
    "baseline": "#4C78A8",
    "rope": "#59A14F",
    "muon": "#F28E2B",
    "untie_embed": "#E15759",
}
METRIC_LABELS = {
    "H_peak": "H_peak",
    "pca_peak_mass_pc1": "PC1 Peak Mass",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _choose_probe_type(df: pd.DataFrame, requested: str) -> str:
    mode = str(requested).strip()
    if mode and mode != "auto":
        return mode
    if df.empty or "probe_type" not in df.columns:
        return "clean_band"
    vals = {str(x).strip() for x in df["probe_type"].dropna().unique()}
    if "clean_band" in vals:
        return "clean_band"
    if "matched_band" in vals:
        return "matched_band"
    return "clean_band"


def _choose_band(df: pd.DataFrame, requested: str) -> str:
    band = str(requested).strip()
    if band:
        return band
    if df.empty or "probe_band" not in df.columns:
        return "97:200"
    vals = sorted(str(x).strip() for x in df["probe_band"].dropna().unique())
    if "97:200" in vals:
        return "97:200"
    return vals[0] if vals else "97:200"


def _load_seq_len(run_dir: str, cache: Dict[str, int]) -> int:
    key = str(run_dir)
    if key in cache:
        return cache[key]
    cfg_path = Path(run_dir) / "run_config.json"
    seq_len = 64
    try:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            seq_len = int(cfg.get("seq_len", 64))
    except Exception:
        seq_len = 64
    cache[key] = seq_len
    return seq_len


def _add_token_coordinate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seq_len_cache: Dict[str, int] = {}

    def _tokens(row: pd.Series) -> float:
        ckpt = pd.to_numeric(pd.Series([row.get("checkpoint", np.nan)]), errors="coerce").iloc[0]
        batch = pd.to_numeric(pd.Series([row.get("B", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(ckpt) or not np.isfinite(batch):
            return float("nan")
        seq_len = _load_seq_len(str(row.get("run_dir", "")), seq_len_cache)
        return float(int(ckpt) * int(batch) * int(seq_len))

    out["tokens_at_checkpoint"] = out.apply(_tokens, axis=1)
    out["checkpoint_progress_pct"] = pd.to_numeric(out.get("checkpoint_progress_pct", np.nan), errors="coerce")
    out["checkpoint"] = pd.to_numeric(out.get("checkpoint", np.nan), errors="coerce")
    return out


def _variant_order(df: pd.DataFrame, requested: str) -> List[str]:
    order = [x.strip() for x in str(requested).split(",") if x.strip()]
    if order:
        return order
    vals = [str(x).strip() for x in df.get("variant_stage", pd.Series(dtype=str)).dropna().unique()]
    keep = [v for v in DEFAULT_VARIANT_ORDER if v in vals]
    rest = [v for v in vals if v not in keep]
    return keep + sorted(rest)


def _plot_metric(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, x_label: str, variant_order: List[str]) -> None:
    plotted = False
    for variant in variant_order:
        sub = df[df["variant_stage"] == variant].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)
        xs = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() == 0:
            continue
        xs = xs[mask]
        ys = ys[mask]
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=4,
            lw=2,
            color=DEFAULT_COLORS.get(variant, None),
            label=variant,
        )
        ax.scatter(xs[-1], ys[-1], s=32, color=DEFAULT_COLORS.get(variant, None), zorder=3)
        plotted = True

    if not plotted:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No finite values for {y_col}.", ha="center", va="center")
        return

    ax.set_xlabel(x_label)
    ax.set_ylabel(METRIC_LABELS.get(y_col, y_col))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def _build_plot(df: pd.DataFrame, x_col: str, x_label: str, out_path: Path, title_prefix: str, variant_order: List[str]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _plot_metric(axes[0], df, x_col, "H_peak", x_label, variant_order)
    _plot_metric(axes[1], df, x_col, "pca_peak_mass_pc1", x_label, variant_order)
    if x_col == "tokens_at_checkpoint":
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}"))
    axes[0].set_title(f"{title_prefix}: H_peak")
    axes[1].set_title(f"{title_prefix}: PC1 Peak Mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean feature-learning curves on progress/token axes.")
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--probe-type", type=str, default="auto")
    parser.add_argument("--band", type=str, default="")
    parser.add_argument("--variant-order", type=str, default="baseline,rope,muon,untie_embed")
    parser.add_argument("--x-axes", type=str, default="progress,tokens")
    parser.add_argument("--out-prefix", type=str, default="")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    summary = _read_csv(feature_dir / "feature_learning_summary.csv")
    if summary.empty:
        raise SystemExit(f"No usable summary CSV found in {feature_dir}")

    probe_type = _choose_probe_type(summary, args.probe_type)
    band = _choose_band(summary, args.band)
    work = summary.copy()
    if "probe_type" in work.columns:
        sub = work[work["probe_type"] == probe_type].copy()
        if not sub.empty:
            work = sub
    if "probe_band" in work.columns:
        sub = work[work["probe_band"] == band].copy()
        if not sub.empty:
            work = sub

    work = _add_token_coordinate(work)
    variant_order = _variant_order(work, args.variant_order)
    out_prefix = Path(args.out_prefix) if args.out_prefix else (feature_dir / "feature_learning_clean")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "variant_stage",
        "variant_index",
        "B",
        "checkpoint",
        "checkpoint_progress_pct",
        "tokens_at_checkpoint",
        "probe_band",
        "probe_type",
        "H_peak",
        "pca_peak_mass_pc1",
        "base_probe_loss",
        "run_name",
        "run_dir",
    ]
    work.reindex(columns=export_cols).to_csv(out_prefix.parent / f"{out_prefix.name}_data.csv", index=False)

    axes_requested = [x.strip() for x in str(args.x_axes).split(",") if x.strip()]
    for axis in axes_requested:
        if axis == "progress":
            _build_plot(
                df=work,
                x_col="checkpoint_progress_pct",
                x_label="Training Progress (%)",
                out_path=out_prefix.parent / f"{out_prefix.name}_progress.png",
                title_prefix=f"{probe_type} {band}",
                variant_order=variant_order,
            )
        elif axis == "tokens":
            _build_plot(
                df=work,
                x_col="tokens_at_checkpoint",
                x_label="Tokens Seen (M)",
                out_path=out_prefix.parent / f"{out_prefix.name}_tokens.png",
                title_prefix=f"{probe_type} {band}",
                variant_order=variant_order,
            )
        elif axis == "checkpoint":
            _build_plot(
                df=work,
                x_col="checkpoint",
                x_label="Checkpoint Step",
                out_path=out_prefix.parent / f"{out_prefix.name}_checkpoint.png",
                title_prefix=f"{probe_type} {band}",
                variant_order=variant_order,
            )
        else:
            raise SystemExit(f"Unknown x-axis mode: {axis}")

    print(f"Wrote: {out_prefix.parent / f'{out_prefix.name}_data.csv'}")
    for axis in axes_requested:
        print(f"Wrote: {out_prefix.parent / f'{out_prefix.name}_{axis}.png'}")


if __name__ == "__main__":
    main()
