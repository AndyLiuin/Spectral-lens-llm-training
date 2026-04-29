from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .runner import train_toy_run
except ImportError:
    from config import MeasurementConfig, RunConfig
    from runner import train_toy_run


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _measurement_from_json(obj: dict) -> MeasurementConfig:
    return MeasurementConfig(
        n_samples=int(obj.get("n_samples", 512)),
        fixed_samples=bool(obj.get("fixed_samples", True)),
        trace_normalize=bool(obj.get("trace_normalize", True)),
        alpha_head_range=tuple(int(x) for x in obj.get("alpha_head_range", [1, 10])),
        alpha_tail_range=tuple(int(x) for x in obj.get("alpha_tail_range", [50, 200])),
    )


def _load_run_config(run_dir: Path) -> RunConfig:
    obj = json.loads((run_dir / "run_config.json").read_text())
    obj["measurement"] = _measurement_from_json(obj.get("measurement", {}))
    obj["param_spectrum_paths"] = tuple(str(x) for x in obj.get("param_spectrum_paths", []))
    obj["checkpoint_steps"] = tuple(int(x) for x in obj.get("checkpoint_steps", []))
    obj["output_root"] = Path(obj.get("output_root", "toy_outputs"))
    return RunConfig(**obj)


def _load_source_final_step(run_dir: Path) -> int:
    summary_csv = run_dir / "run_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if not df.empty:
            value = pd.to_numeric(df.iloc[0].get("final_step"), errors="coerce")
            if not pd.isna(value):
                return int(value)
    summary_json = run_dir / "run_summary.json"
    if summary_json.exists():
        obj = json.loads(summary_json.read_text())
        value = pd.to_numeric(obj.get("final_step"), errors="coerce")
        if not pd.isna(value):
            return int(value)
    metrics_csv = run_dir / "metrics_over_time.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if not df.empty and "step" in df.columns:
            value = pd.to_numeric(df["step"], errors="coerce").dropna().max()
            if not pd.isna(value):
                return int(value)
    raise FileNotFoundError(f"Could not determine final_step from {run_dir}")


def _checkpoint_steps_from_reference(reference_step: int, fractions: List[float]) -> Tuple[int, ...]:
    if int(reference_step) <= 1:
        return ()
    steps: List[int] = []
    for frac in fractions:
        step = int(round(float(reference_step) * float(frac)))
        step = max(1, min(int(reference_step) - 1, step))
        steps.append(step)
    return tuple(sorted(set(steps)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one selected run with an explicit checkpoint schedule.")
    parser.add_argument("--source-run-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--ablation-name", type=str, required=True)
    parser.add_argument("--checkpoint-fractions", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--write-summary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    source_run_dir = Path(args.source_run_dir)
    out_root = Path(args.out_root)
    checkpoint_fractions = parse_float_list(args.checkpoint_fractions)
    for frac in checkpoint_fractions:
        if not (0.0 < float(frac) < 1.0):
            raise SystemExit(f"--checkpoint-fractions entries must lie strictly between 0 and 1, got: {frac}")

    cfg = _load_run_config(source_run_dir)
    source_final_step = _load_source_final_step(source_run_dir)
    checkpoint_steps = _checkpoint_steps_from_reference(source_final_step, checkpoint_fractions)
    replay_cfg = replace(
        cfg,
        output_root=out_root,
        save_checkpoints=True,
        checkpoint_every=0,
        checkpoint_steps=checkpoint_steps,
    )

    print(
        "[single-replay] "
        f"variant={cfg.variant} "
        f"source_final_step={source_final_step} "
        f"checkpoint_steps={','.join(str(x) for x in checkpoint_steps)} "
        f"device={args.device or 'auto'}",
        flush=True,
    )
    if args.dry_run:
        print(f"[single-replay] run_name={replay_cfg.run_name()}", flush=True)
        print(f"[single-replay] out_root={out_root}", flush=True)
        print(f"[single-replay] ablation_name={args.ablation_name}", flush=True)
        return

    state = train_toy_run(
        config=replay_cfg,
        ablation_name=args.ablation_name,
        write_summary=bool(args.write_summary),
        save_spectra=True,
        device=args.device,
    )
    print(f"[single-replay] completed run_dir={state.run_dir}", flush=True)


if __name__ == "__main__":
    main()
