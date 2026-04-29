from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_module(*candidates: str) -> str:
    for name in candidates:
        if importlib.util.find_spec(name) is not None:
            return name
    raise ModuleNotFoundError(f"Could not resolve any module from: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness-first toy feature-learning pipeline.")
    parser.add_argument("--ablation-dir", type=str, default="toy_model/concat_batch_regime_selected_runs/variant_concat_ablation")
    parser.add_argument("--feature-dir", type=str, default="")
    parser.add_argument("--run-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tracks", type=str, default="a")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--variant-order", type=str, default="baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale")
    parser.add_argument("--B-list", type=str, default="32,128,512")
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--lr-multipliers", type=str, default="0.5,0.707,1.0,1.414,2.0")
    parser.add_argument("--lr-scaling", type=str, default="both", choices=["sqrt", "linear", "both", "none"])
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--target-loss", type=float, default=3.0)
    parser.add_argument("--target-loss-metric", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--target-loss-patience", type=int, default=1)
    parser.add_argument("--target-loss-min-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--measurement-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--bands", type=str, default="97:200,179:50")
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--checkpoint-fracs", type=str, default="")
    parser.add_argument("--alignment-mode", type=str, default="step", choices=["step", "progress"])
    parser.add_argument("--checkpoint-progress-pcts", type=str, default="")
    parser.add_argument("--early-checkpoints", type=str, default="400,800,1600")
    parser.add_argument("--early-progress-pcts", type=str, default="")
    parser.add_argument("--taxonomy-checkpoint", type=int, default=1600)
    parser.add_argument("--taxonomy-progress-pct", type=int, default=50)
    parser.add_argument("--num-loss-proximal-targets", type=int, default=5)
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--probe-target-loss", type=float, default=None)
    parser.add_argument("--probe-target-loss-metric", type=str, default="", choices=["", "val", "train"])
    parser.add_argument("--save-param-spectra", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-svd-samples", type=int, default=64)
    parser.add_argument("--run-bridge-analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    py = sys.executable
    ablation_dir = Path(args.ablation_dir)
    feature_dir = Path(args.feature_dir) if args.feature_dir else (ablation_dir / "feature_learning_analysis")
    selected_out_root = ablation_dir.parent
    out_root = selected_out_root.parent if selected_out_root.name == "concat_batch_regime_selected_runs" else selected_out_root
    train_module = _resolve_module("run_concat_batch_regime_ablation", "toy_model.run_concat_batch_regime_ablation")
    probe_module = _resolve_module("run_feature_learning_probe", "toy_model.run_feature_learning_probe")
    map_module = _resolve_module("analyze_feature_learning_variant_map", "toy_model.analyze_feature_learning_variant_map")
    bridge_module = _resolve_module("old_scripts.analyze_feature_learning_bridge", "toy_model.old_scripts.analyze_feature_learning_bridge")
    plot_module = _resolve_module("plot_feature_learning_panels", "toy_model.plot_feature_learning_panels")

    if args.run_training:
        train_cmd = [
            py,
            "-m",
            train_module,
            "--task",
            "mod_arith_lm",
            "--tracks",
            args.tracks,
            "--variant-order",
            args.variant_order,
            "--B-list",
            args.B_list,
            "--base-lr",
            str(args.base_lr),
            "--lr-multipliers",
            args.lr_multipliers,
            "--lr-scaling",
            args.lr_scaling,
            "--seeds",
            args.seeds,
            "--max-steps",
            str(args.max_steps),
            "--log-every",
            str(args.log_every),
            "--eval-every",
            str(args.eval_every),
            "--measurement-every",
            str(args.measurement_every),
            "--target-loss",
            str(args.target_loss),
            "--target-loss-metric",
            args.target_loss_metric,
            "--target-loss-patience",
            str(args.target_loss_patience),
            "--target-loss-min-steps",
            str(args.target_loss_min_steps),
            "--materialize-selected-runs",
            "--save-checkpoints",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--save-param-spectra" if args.save_param_spectra else "--no-save-param-spectra",
            "--grad-svd-samples",
            str(args.grad_svd_samples),
            "--out-root",
            str(out_root),
            "--selected-out-root",
            str(selected_out_root),
            "--selected-ablation-name",
            ablation_dir.name,
            "--probe-regime",
            args.probe_regime,
        ]
        if args.device:
            train_cmd.extend(["--device", args.device])
        _run(train_cmd)

    probe_cmd = [
        py,
        "-m",
        probe_module,
        "--ablation-dir",
        str(ablation_dir),
        "--bands",
        args.bands,
        "--probe-regime",
        args.probe_regime,
        "--out-dir",
        str(feature_dir),
    ]
    if str(args.checkpoints).strip():
        probe_cmd.extend(["--checkpoints", args.checkpoints])
    if str(args.checkpoint_fracs).strip():
        probe_cmd.extend(["--checkpoint-fracs", args.checkpoint_fracs])
    if args.probe_target_loss is not None:
        probe_cmd.extend(["--target-loss", str(args.probe_target_loss)])
    if str(args.probe_target_loss_metric).strip():
        probe_cmd.extend(["--target-loss-metric", args.probe_target_loss_metric])
    if args.device:
        probe_cmd.extend(["--device", args.device])
    _run(probe_cmd)

    map_cmd = [
        py,
        "-m",
        map_module,
        "--ablation-dir",
        str(ablation_dir),
        "--feature-dir",
        str(feature_dir),
        "--bands",
        args.bands,
        "--alignment-mode",
        args.alignment_mode,
        "--early-checkpoints",
        args.early_checkpoints,
        "--early-progress-pcts",
        args.early_progress_pcts,
        "--taxonomy-checkpoint",
        str(args.taxonomy_checkpoint),
        "--taxonomy-progress-pct",
        str(args.taxonomy_progress_pct),
        "--num-loss-proximal-targets",
        str(args.num_loss_proximal_targets),
        "--out-dir",
        str(feature_dir),
    ]
    if str(args.checkpoints).strip():
        map_cmd.extend(["--checkpoints", args.checkpoints])
    if str(args.checkpoint_progress_pcts).strip():
        map_cmd.extend(["--checkpoint-progress-pcts", args.checkpoint_progress_pcts])
    _run(map_cmd)

    if args.run_bridge_analysis:
        _run(
            [
                py,
                "-m",
                bridge_module,
                "--ablation-dir",
                str(ablation_dir),
                "--feature-analysis-dir",
                str(feature_dir),
                "--out-dir",
                str(feature_dir),
            ]
        )

    _run(
        [
            py,
            "-m",
            plot_module,
            "--feature-dir",
            str(feature_dir),
        ]
    )

    print("Feature-learning upgrade pipeline complete.")


if __name__ == "__main__":
    main()
