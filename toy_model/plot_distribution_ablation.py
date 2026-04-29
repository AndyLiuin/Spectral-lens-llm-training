from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distribution-ablation summary.")
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--in-file", type=str, default="")
    parser.add_argument("--out-file", type=str, default="")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    in_file = Path(args.in_file) if args.in_file else out_root / "distribution_ablation" / "distribution_ablation_aggregate.csv"
    out_file = Path(args.out_file) if args.out_file else out_root / "distribution_ablation" / "distribution_ablation_plot.png"

    if not in_file.exists():
        raise FileNotFoundError(f"Missing aggregate file: {in_file}")

    df = pd.read_csv(in_file)
    if df.empty:
        raise ValueError("Aggregate distribution-ablation file is empty.")

    df["condition"] = (
        df["latent_dist"].astype(str)
        + "|"
        + df["latent_anisotropy"].astype(str)
        + "|g="
        + df["latent_anisotropy_gamma"].astype(str)
    )

    tracks = sorted(df["track"].unique())
    fig, axes = plt.subplots(len(tracks), 2, figsize=(14, 4 * len(tracks)), squeeze=False)

    for i, track in enumerate(tracks):
        sub = df[df["track"] == track].copy()
        sub = sub.sort_values(["latent_dist", "latent_anisotropy", "latent_anisotropy_gamma"])

        ax1 = axes[i, 0]
        ax1.bar(sub["condition"], sub["alpha_tail_mean"], color="#2c7fb8")
        ax1.set_title(f"Track {track}: alpha_tail across data distributions")
        ax1.set_ylabel("alpha_tail_mean")
        ax1.tick_params(axis="x", rotation=35)

        ax2 = axes[i, 1]
        ax2.bar(sub["condition"], sub["rankme_mean"], color="#f03b20")
        ax2.set_title(f"Track {track}: RankMe across data distributions")
        ax2.set_ylabel("rankme_mean")
        ax2.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=220)
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
