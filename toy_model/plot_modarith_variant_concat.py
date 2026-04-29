from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .metrics import compute_alpha, compute_rankme, normalize_trace


def fit_powerlaw_segment(spectrum: np.ndarray, start_idx: int, end_idx: int) -> float:
    s = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    start_idx = max(0, int(start_idx))
    end_idx = min(len(s), int(end_idx))
    if end_idx - start_idx < 2:
        return float("nan")
    y_seg = s[start_idx:end_idx]
    x_seg = np.arange(start_idx + 1, end_idx + 1, dtype=np.float64)
    mask = y_seg > 1e-20
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(x_seg[mask]), np.log(y_seg[mask]), 1)
    return float(-slope)


def load_spectrum_series(
    run_dir: Path,
    steps: list[int],
    kind: str = "cov",
    matrix_name: Optional[str] = None,
    normalize: bool = True,
) -> list[np.ndarray]:
    out = []
    for step in steps:
        suffix = f"__{matrix_name}" if matrix_name else ""
        path = run_dir / "spectra" / f"{kind}_spectrum_step_{int(step):06d}{suffix}.npy"
        if not path.exists():
            out.append(np.array([], dtype=np.float64))
            continue
        spec = np.load(path)
        out.append(normalize_trace(spec) if normalize else np.asarray(spec, dtype=np.float64))
    return out


def build_dynamics_table(
    metrics_df: pd.DataFrame,
    act_spectra: list[np.ndarray],
    grad_spectra: list[np.ndarray],
    alpha1: tuple[int, int],
    alpha2: tuple[int, int],
) -> pd.DataFrame:
    rows = []
    for (_, row), act_spec, grad_spec in zip(metrics_df.iterrows(), act_spectra, grad_spectra):
        rows.append(
            {
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]) if "train_loss" in row else np.nan,
                "val_loss": float(row["loss"]),
                "rankme": float(row["rankme"]) if "rankme" in row else compute_rankme(act_spec),
                "alpha1": fit_powerlaw_segment(act_spec, alpha1[0], alpha1[1]),
                "alpha2": fit_powerlaw_segment(act_spec, alpha2[0], alpha2[1]),
                "alpha_head_default": float(row["alpha_head"]) if "alpha_head" in row else compute_alpha(act_spec, 1, 10),
                "alpha_tail_default": float(row["alpha_tail"]) if "alpha_tail" in row else compute_alpha(act_spec, 50, 200),
                "grad_rankme": float(row["grad_rankme"]) if "grad_rankme" in row else compute_rankme(grad_spec),
                "grad_alpha1": fit_powerlaw_segment(grad_spec, alpha1[0], alpha1[1]),
                "grad_alpha2": fit_powerlaw_segment(grad_spec, alpha2[0], alpha2[1]),
                "grad_alpha_head_default": float(row["grad_alpha_head"]) if "grad_alpha_head" in row else compute_alpha(grad_spec, 1, 10),
                "grad_alpha_tail_default": float(row["grad_alpha_tail"]) if "grad_alpha_tail" in row else compute_alpha(grad_spec, 50, 200),
                "tokens_seen": float(row["tokens_seen"]) if "tokens_seen" in row else np.nan,
                "train_time_s": float(row["train_time_s"]) if "train_time_s" in row else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_overview_plot(dyn: pd.DataFrame, out_path: Path, alpha1: tuple[int, int], alpha2: tuple[int, int]) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    ax[0, 0].plot(dyn["step"], dyn["train_loss"], label="Train Loss", marker=".", alpha=0.7)
    ax[0, 0].plot(dyn["step"], dyn["val_loss"], label="Val Loss", marker=".")
    ax[0, 0].set_title("Loss Evolution")
    ax[0, 0].set_xlabel("Step")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(dyn["step"], dyn["alpha1"], label=f"Act Alpha [{alpha1[0]+1},{alpha1[1]}]", marker=".")
    ax[0, 1].plot(dyn["step"], dyn["alpha2"], label=f"Act Alpha [{alpha2[0]+1},{alpha2[1]}]", marker=".")
    ax[0, 1].set_title("Activation Spectrum Alpha")
    ax[0, 1].set_xlabel("Step")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    x_tokens = dyn["tokens_seen"].to_numpy(dtype=float)
    if np.isfinite(x_tokens).any() and np.nanmax(x_tokens) > 0:
        ax[0, 2].plot(x_tokens, dyn["rankme"], label="Act RankMe", color="purple", marker="x")
        ax[0, 2].set_xlabel("Tokens")
    else:
        ax[0, 2].plot(dyn["step"], dyn["rankme"], label="Act RankMe", color="purple", marker="x")
        ax[0, 2].set_xlabel("Step")
    ax[0, 2].set_title("Activation RankMe")
    ax[0, 2].set_ylabel("RankMe")
    ax[0, 2].legend()
    ax[0, 2].grid(True, alpha=0.3)

    ax[1, 0].plot(dyn["step"], dyn["grad_alpha1"], label=f"Grad Alpha [{alpha1[0]+1},{alpha1[1]}]", marker=".")
    ax[1, 0].plot(dyn["step"], dyn["grad_alpha2"], label=f"Grad Alpha [{alpha2[0]+1},{alpha2[1]}]", marker=".")
    ax[1, 0].set_title("Gradient Spectrum Alpha")
    ax[1, 0].set_xlabel("Step")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)

    if np.isfinite(x_tokens).any() and np.nanmax(x_tokens) > 0:
        ax[1, 1].plot(x_tokens, dyn["grad_rankme"], label="Grad RankMe", color="teal", marker="x")
        ax[1, 1].set_xlabel("Tokens")
    else:
        ax[1, 1].plot(dyn["step"], dyn["grad_rankme"], label="Grad RankMe", color="teal", marker="x")
        ax[1, 1].set_xlabel("Step")
    ax[1, 1].set_title("Gradient RankMe")
    ax[1, 1].set_ylabel("RankMe")
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3)

    if np.isfinite(dyn["train_time_s"]).any() and float(np.nanmax(dyn["train_time_s"])) > 0:
        ax[1, 2].plot(dyn["step"], dyn["train_time_s"], marker=".")
        ax[1, 2].set_title("Wall Time by Step")
        ax[1, 2].set_ylabel("Train Time (s)")
    else:
        ax[1, 2].plot(dyn["step"], dyn["tokens_seen"], marker=".")
        ax[1, 2].set_title("Tokens by Step")
        ax[1, 2].set_ylabel("Tokens Seen")
    ax[1, 2].set_xlabel("Step")
    ax[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_rankme_log_plot(dyn: pd.DataFrame, out_path: Path) -> None:
    x_tokens = dyn["tokens_seen"].to_numpy(dtype=float)
    if not (np.isfinite(x_tokens).any() and np.nanmax(x_tokens) > 0):
        return
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_tokens, dyn["rankme"], label="RankMe", color="purple", marker="x")
    plt.xscale("log")
    plt.title("RankMe vs Tokens (Log Scale)")
    plt.xlabel("Tokens Digested (Log Scale)")
    plt.ylabel("RankMe")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_time_plot(dyn: pd.DataFrame, out_path: Path, alpha1: tuple[int, int], alpha2: tuple[int, int]) -> None:
    x_time = dyn["train_time_s"].to_numpy(dtype=float)
    if not (np.isfinite(x_time).any() and np.nanmax(x_time) > 0):
        return
    fig = plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_time, dyn["train_loss"], linewidth=1.0, label="train", alpha=0.6)
    plt.plot(x_time, dyn["val_loss"], marker="o", markersize=4, linewidth=1.5, label="val", color="red")
    plt.title("Loss vs Train Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x_time, dyn["alpha1"], marker="o", markersize=3, label=f"Ranks {alpha1[0]+1}-{alpha1[1]}")
    plt.plot(x_time, dyn["alpha2"], marker="s", markersize=3, label=f"Ranks {alpha2[0]+1}-{alpha2[1]}")
    plt.title("Alpha Evolution")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Power Law Slope (Alpha)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x_time, dyn["rankme"], marker="x", color="purple", label="RankMe")
    plt.title("RankMe vs Train Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("RankMe")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_spectrum_plot(
    dyn: pd.DataFrame,
    spectra: list[np.ndarray],
    out_path: Path,
    title: str = "Spectrum Evolution",
    y_label: str = "Eigenvalue",
) -> None:
    valid = [(int(step), spec) for step, spec in zip(dyn["step"], spectra) if len(spec) > 0]
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.plasma(np.linspace(0, 1, len(valid)))
    for i, (step, spec) in enumerate(valid):
        is_boundary = i == 0 or i == len(valid) - 1
        ax.loglog(
            np.arange(1, len(spec) + 1),
            spec,
            color=colors[i],
            alpha=1.0 if is_boundary else 0.5,
            linewidth=2.5 if is_boundary else 1.0,
            label=f"Step {step}" if is_boundary else None,
        )
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel(y_label)
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=valid[0][0], vmax=valid[-1][0]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Training Step")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_spectrum_gif(dyn: pd.DataFrame, spectra: list[np.ndarray], out_path: Path, alpha1: tuple[int, int], alpha2: tuple[int, int]) -> None:
    valid = [(row, spec) for (_, row), spec in zip(dyn.iterrows(), spectra) if len(spec) > 0]
    if len(valid) < 2:
        return

    stacked = np.stack([spec for _, spec in valid], axis=0)
    steps = [int(row["step"]) for row, _ in valid]
    times = [float(row["train_time_s"]) for row, _ in valid]
    a1_vals = [float(row["alpha1"]) for row, _ in valid]
    a2_vals = [float(row["alpha2"]) for row, _ in valid]

    positive = stacked[stacked > 0]
    ymin = max(float(positive.min()) if positive.size else 1e-18, 1e-18)
    ymax = float(stacked.max()) if stacked.size else 1.0
    ranks = np.arange(1, stacked.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, stacked.shape[1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Spectrum Evolution")

    (line,) = ax.plot(ranks, stacked[0], lw=2, color="blue")
    text = ax.text(0.05, 0.1, "", transform=ax.transAxes, va="bottom", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    def update(frame_idx: int):
        line.set_data(ranks, stacked[frame_idx])
        text.set_text(
            f"Step: {steps[frame_idx]}\n"
            f"Time: {times[frame_idx]:.2f}s\n"
            f"Alpha ({alpha1[0]+1}-{alpha1[1]}): {a1_vals[frame_idx]:.3f}\n"
            f"Alpha ({alpha2[0]+1}-{alpha2[1]}): {a2_vals[frame_idx]:.3f}"
        )
        return (line, text)

    ani = FuncAnimation(fig, update, frames=len(valid), interval=120, blit=True)
    ani.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def save_endpoint_plot(summary_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["variant_index", "variant_stage", "variant_combo"], as_index=False)
        .agg(
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            rankme_mean=("rankme", "mean"),
            rankme_std=("rankme", "std"),
            alpha_tail_mean=("alpha_tail", "mean"),
            alpha_tail_std=("alpha_tail", "std"),
            n=("loss", "size"),
        )
        .sort_values("variant_index")
    )

    labels = grouped["variant_combo"].tolist()
    x = np.arange(len(grouped))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].errorbar(x, grouped["test_loss_mean"], yerr=grouped["test_loss_std"].fillna(0.0), marker="o")
    axes[0].set_title("Endpoint Test Loss")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(x, grouped["alpha_tail_mean"], yerr=grouped["alpha_tail_std"].fillna(0.0), marker="o")
    axes[1].set_title("Endpoint Alpha Tail")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(x, grouped["rankme_mean"], yerr=grouped["rankme_std"].fillna(0.0), marker="o", color="purple")
    axes[2].set_title("Endpoint RankMe")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return grouped


def discover_matrix_names(run_dir: Path, max_matrices: int) -> list[str]:
    csv_path = run_dir / "param_spectra_over_time.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty or "matrix_name" not in df.columns:
        return []
    names = [str(x) for x in df["matrix_name"].dropna().tolist() if str(x)]
    ordered_unique: list[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered_unique.append(name)
    if max_matrices > 0:
        return ordered_unique[:max_matrices]
    return ordered_unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and analyze cumulative modarith variant runs.")
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--ablation", type=str, default="variant_concat_ablation")
    parser.add_argument("--track", type=str, default="a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha1", type=str, default="0,5")
    parser.add_argument("--alpha2", type=str, default="9,50")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-matrices", type=int, default=4)
    parser.add_argument("--make-gif", action="store_true")
    args = parser.parse_args()

    alpha1 = tuple(int(x.strip()) for x in args.alpha1.split(","))
    alpha2 = tuple(int(x.strip()) for x in args.alpha2.split(","))

    out_root = Path(args.out_root)
    ablation_dir = out_root / args.ablation
    summary_path = ablation_dir / "variant_concat_ablation_summary.csv"
    summary_df = pd.read_csv(summary_path)

    analysis_dir = ablation_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    endpoint_df = save_endpoint_plot(summary_df, analysis_dir / "concat_endpoint_summary.png")
    endpoint_df.to_csv(analysis_dir / "concat_endpoint_summary.csv", index=False)

    filtered = summary_df[(summary_df["track"] == args.track) & (summary_df["seed"] == args.seed)].sort_values("variant_index")
    if args.max_runs > 0:
        filtered = filtered.head(args.max_runs)

    manifest_rows = []
    for _, row in filtered.iterrows():
        run_dir = Path(row["run_dir"])
        metrics_path = run_dir / "metrics_over_time.csv"
        if not metrics_path.exists():
            continue
        metrics_df = pd.read_csv(metrics_path).sort_values("step")
        steps = metrics_df["step"].astype(int).tolist()
        act_spectra = load_spectrum_series(run_dir, steps, kind="cov", normalize=True)
        grad_spectra = load_spectrum_series(run_dir, steps, kind="grad", normalize=True)
        dyn = build_dynamics_table(metrics_df, act_spectra, grad_spectra, alpha1=alpha1, alpha2=alpha2)

        base = analysis_dir / f"{row['variant_index']:02d}_{row['variant_stage']}"
        dyn.to_csv(base.with_name(base.name + "_dynamics.csv"), index=False)
        save_overview_plot(dyn, base.with_name(base.name + "_overview.png"), alpha1=alpha1, alpha2=alpha2)
        save_rankme_log_plot(dyn, base.with_name(base.name + "_rankme_log_tokens.png"))
        save_time_plot(dyn, base.with_name(base.name + "_time_overview.png"), alpha1=alpha1, alpha2=alpha2)
        save_spectrum_plot(
            dyn,
            act_spectra,
            base.with_name(base.name + "_act_spectrum_evolution.png"),
            title="Activation Covariance Spectrum Evolution",
            y_label="Eigenvalue",
        )
        save_spectrum_plot(
            dyn,
            grad_spectra,
            base.with_name(base.name + "_grad_repr_spectrum_evolution.png"),
            title="Gradient-Proxy Spectrum Evolution",
            y_label="Eigenvalue",
        )
        if args.make_gif:
            save_spectrum_gif(
                dyn,
                act_spectra,
                base.with_name(base.name + "_act_spectrum_evolution.gif"),
                alpha1=alpha1,
                alpha2=alpha2,
            )

        matrix_names = discover_matrix_names(run_dir=run_dir, max_matrices=args.max_matrices)
        for matrix_name in matrix_names:
            gradsvd_spectra = load_spectrum_series(
                run_dir,
                steps,
                kind="gradsvd",
                matrix_name=matrix_name,
                normalize=False,
            )
            weight_spectra = load_spectrum_series(
                run_dir,
                steps,
                kind="weight",
                matrix_name=matrix_name,
                normalize=False,
            )
            save_spectrum_plot(
                dyn,
                gradsvd_spectra,
                base.with_name(base.name + f"_gradsvd_{matrix_name}.png"),
                title=f"Gradient SVD Spectrum ({matrix_name})",
                y_label="Singular value",
            )
            save_spectrum_plot(
                dyn,
                weight_spectra,
                base.with_name(base.name + f"_weightsvd_{matrix_name}.png"),
                title=f"Weight Spectrum ({matrix_name})",
                y_label="Singular value",
            )

        manifest_rows.append(
            {
                "variant_index": int(row["variant_index"]),
                "variant_stage": row["variant_stage"],
                "variant_combo": row["variant_combo"],
                "run_dir": str(run_dir),
                "dynamics_csv": str(base.with_name(base.name + "_dynamics.csv")),
                "overview_png": str(base.with_name(base.name + "_overview.png")),
                "time_overview_png": str(base.with_name(base.name + "_time_overview.png")),
                "rankme_log_png": str(base.with_name(base.name + "_rankme_log_tokens.png")),
                "act_spectrum_png": str(base.with_name(base.name + "_act_spectrum_evolution.png")),
                "grad_repr_spectrum_png": str(base.with_name(base.name + "_grad_repr_spectrum_evolution.png")),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(analysis_dir / "analysis_manifest.csv", index=False)
    print(f"Wrote: {analysis_dir / 'concat_endpoint_summary.csv'}")
    print(f"Wrote: {analysis_dir / 'analysis_manifest.csv'}")


if __name__ == "__main__":
    main()
