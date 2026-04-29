from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all toy-model ablation scripts in sequence.")
    parser.add_argument("--out-root", type=str, default="toy_outputs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--task", type=str, default="rff_regression", choices=["rff_regression", "mod_arith_lm"])
    parser.add_argument("--seq-len", type=int, default=16)
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
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--embed-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--scalar-lr", type=float, default=None)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument(
        "--modarith-measurement-pooling",
        type=str,
        default="last",
        choices=["token", "last", "mean"],
    )
    parser.add_argument("--fast", action="store_true", help="Use reduced smoke-style settings.")
    parser.add_argument(
        "--include-concat-batch-regime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the cumulative-stage batch/LR sweep pipeline.",
    )
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    common = [
        "--out-root",
        str(root),
        "--device",
        args.device,
        "--task",
        args.task,
        "--seq-len",
        str(args.seq_len),
        "--vocab-size",
        str(args.vocab_size),
        "--zipf-c",
        str(args.zipf_c),
        "--zipf-o",
        str(args.zipf_o),
        "--c-min",
        str(args.c_min),
        "--min-step-frac",
        str(args.min_step_frac),
        "--noncoprime-prob",
        str(args.noncoprime_prob),
        "--mix-components-min",
        str(args.mix_components_min),
        "--mix-components-max",
        str(args.mix_components_max),
        "--component-weight-pareto-alpha",
        str(args.component_weight_pareto_alpha),
        "--token-noise-std",
        str(args.token_noise_std),
        "--token-noise-t-df",
        str(args.token_noise_t_df),
        "--probe-regime",
        args.probe_regime,
        "--modarith-measurement-pooling",
        args.modarith_measurement_pooling,
        "--allow-noncoprime" if args.allow_noncoprime else "--no-allow-noncoprime",
    ]
    if args.embed_lr is not None:
        common.extend(["--embed-lr", str(args.embed_lr)])
    if args.head_lr is not None:
        common.extend(["--head-lr", str(args.head_lr)])
    if args.scalar_lr is not None:
        common.extend(["--scalar-lr", str(args.scalar_lr)])
    if args.muon_lr is not None:
        common.extend(["--muon-lr", str(args.muon_lr)])
    common.extend(["--muon-momentum", str(args.muon_momentum)])

    if args.fast:
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_estimator_ablation",
                "--tracks",
                "a,b",
                "--d",
                "4",
                "--beta",
                "1.0",
                "--P",
                "128",
                "--D",
                "4000",
                "--max-steps",
                "200",
                "--n-list",
                "128,256,512",
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_noise_scale_ablation",
                "--tracks",
                "a,b",
                "--d",
                "4",
                "--beta",
                "1.0",
                "--P",
                "128",
                "--D",
                "4000",
                "--B-list",
                "32,128,512",
                "--base-lr",
                "3e-4",
                "--max-steps",
                "250",
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_phase_ablation",
                "--tracks",
                "a,b",
                "--d",
                "4",
                "--beta",
                "1.0",
                "--P",
                "128",
                "--D",
                "4000",
                "--max-steps",
                "300",
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_scaling_link_ablation",
                "--tracks",
                "a,b",
                "--d-values",
                "4,8",
                "--beta-values",
                "1.0,1.5",
                "--P-values",
                "128,256",
                "--D-values",
                "2000,8000,32000",
                "--seeds",
                "0,1",
                "--max-steps",
                "250",
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_variant_ablation",
                "--tracks",
                "a",
                "--variants",
                "baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale",
                "--d",
                "4",
                "--beta",
                "1.0",
                "--P",
                "128",
                "--D",
                "4000",
                "--seeds",
                "0",
                "--max-steps",
                "250",
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_distribution_ablation",
                "--tracks",
                "a,b",
                "--d",
                "4",
                "--beta",
                "1.0",
                "--P",
                "128",
                "--D",
                "4000",
                "--seeds",
                "0",
                "--max-steps",
                "200",
                "--latent-dists",
                "gaussian,uniform,student_t",
                "--anisotropy-modes",
                "isotropic,powerlaw",
                "--anisotropy-gammas",
                "1.0",
                *common,
            ]
        )
        if args.include_concat_batch_regime:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "toy_model.run_concat_batch_regime_ablation",
                    "--tracks",
                    "a",
                    "--task",
                    "mod_arith_lm",
                    "--variant-order",
                    "baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale",
                    "--B-list",
                    "32,128",
                    "--base-lr",
                    "3e-4",
                    "--max-steps",
                    "200",
                    "--seeds",
                    "0",
                    *common,
                ]
            )
    else:
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_estimator_ablation",
                "--tracks",
                "a,b",
                "--max-steps",
                str(args.max_steps),
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_noise_scale_ablation",
                "--tracks",
                "a,b",
                "--max-steps",
                str(args.max_steps),
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_phase_ablation",
                "--tracks",
                "a,b",
                "--max-steps",
                str(max(args.max_steps, 1200)),
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_scaling_link_ablation",
                "--tracks",
                "a,b",
                "--max-steps",
                str(args.max_steps),
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_variant_ablation",
                "--tracks",
                "a",
                "--variants",
                "baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale",
                "--max-steps",
                str(args.max_steps),
                *common,
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "toy_model.run_distribution_ablation",
                "--tracks",
                "a,b",
                "--max-steps",
                str(args.max_steps),
                *common,
            ]
        )
        if args.include_concat_batch_regime:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "toy_model.run_concat_batch_regime_ablation",
                    "--tracks",
                    "a",
                    "--variant-order",
                    "baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale",
                    "--B-list",
                    "32,128,512",
                    "--base-lr",
                    "3e-4",
                    "--max-steps",
                    str(args.max_steps),
                    *common,
                ]
            )

    run_cmd(
        [
            sys.executable,
            "-m",
            "toy_model.plot_toy_main_2x2",
            "--out-root",
            str(root),
            "--out-file",
            str(root / "toy_main_2x2.png"),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "toy_model.plot_distribution_ablation",
            "--out-root",
            str(root),
        ]
    )


if __name__ == "__main__":
    main()
