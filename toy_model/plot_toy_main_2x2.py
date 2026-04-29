from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2x2 toy main figure.")
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--out-file", type=str, default="toy_outputs/toy_main_2x2.png")
    args = parser.parse_args()

    out_root = Path(args.out_root)

    estimator_path = out_root / "estimator_ablation" / "estimator_ablation_summary.csv"
    noise_path = out_root / "noise_scale_ablation" / "noise_scale_matched_loss.csv"
    phase_path = out_root / "phase_ablation" / "phase_trajectories.csv"
    fit_path = out_root / "scaling_link_ablation" / "scaling_fit.csv"
    corr_path = out_root / "scaling_link_ablation" / "scaling_correlation.json"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # 1) Estimator panel.
    ax = axes[0, 0]
    df = safe_read_csv(estimator_path)
    if not df.empty:
        for (track, mode), g in df.groupby(["track", "measurement_mode"]):
            g = g.sort_values("d_over_n")
            ax.plot(g["d_over_n"], g["alpha_tail"], marker="o", label=f"{track}-{mode}")
        ax.set_xlabel("d/n")
        ax.set_ylabel(r"$\alpha_{tail}$")
        ax.set_title("Estimator Effect")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Missing estimator data", ha="center", va="center")

    # 2) Noise-scale panel.
    ax = axes[0, 1]
    df = safe_read_csv(noise_path)
    if not df.empty:
        for (track, regime), g in df.groupby(["track", "regime"]):
            g = g.sort_values("matched_loss_target")
            ax.plot(g["matched_loss_target"], g["mean_js_div"], marker="o", label=f"{track}-{regime}")
        ax.set_xlabel("Matched Loss")
        ax.set_ylabel("Mean JS Divergence")
        ax.set_title("Noise-Scale Matched-Loss Divergence")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Missing noise-scale data", ha="center", va="center")

    # 3) Phase panel.
    ax = axes[1, 0]
    df = safe_read_csv(phase_path)
    if not df.empty:
        for (track, regime), g in df.groupby(["track", "regime"]):
            g = g.sort_values("step")
            ax.plot(g["step"], g["rankme"], label=f"{track}-{regime}")
        ax.set_xlabel("Step")
        ax.set_ylabel("RankMe")
        ax.set_title("Phase-Like RankMe Trajectories")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Missing phase data", ha="center", va="center")

    # 4) Scaling-link panel.
    ax = axes[1, 1]
    df = safe_read_csv(fit_path)
    if not df.empty:
        ax.scatter(df["s"], df["alpha_proxy"], alpha=0.8, s=24)
        if len(df) >= 2:
            x = df["s"].to_numpy(dtype=float)
            y = df["alpha_proxy"].to_numpy(dtype=float)
            if np.std(x) > 1e-12:
                m, b = np.polyfit(x, y, deg=1)
                xs = np.linspace(np.min(x), np.max(x), 100)
                ax.plot(xs, m * xs + b, color="black", lw=1.6)
        title = "Scaling-Link"
        if corr_path.exists():
            with corr_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            r = info.get("pearson_r", float("nan"))
            title += f" (r={r:.2f})"
        ax.set_title(title)
        ax.set_xlabel("Data-scaling exponent s")
        ax.set_ylabel(r"Spectral proxy $\alpha_{tail}$")
    else:
        ax.text(0.5, 0.5, "Missing scaling-link data", ha="center", va="center")

    fig.tight_layout()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=220)
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
