from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from .config import MeasurementConfig, RunConfig
    from .run_concat_batch_regime_ablation import (
        build_variant_concat_matched_tables,
        build_variant_concat_summary_row,
    )
    from .runner import train_toy_run
except ImportError:
    from config import MeasurementConfig, RunConfig
    from run_concat_batch_regime_ablation import (
        build_variant_concat_matched_tables,
        build_variant_concat_summary_row,
    )
    from runner import train_toy_run


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _resolve_run_dir(row: pd.Series, ablation_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    run_name = row.get("run_name", "")
    if isinstance(run_name, str) and run_name.strip():
        candidates.append(ablation_dir / run_name.strip())
    run_dir = row.get("run_dir", "")
    if isinstance(run_dir, str) and run_dir.strip():
        p = Path(run_dir.strip())
        candidates.append(p if p.is_absolute() else (Path.cwd() / p))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def _checkpoint_steps_from_reference(reference_step: int, fractions: Sequence[float]) -> Tuple[int, ...]:
    if int(reference_step) <= 1:
        return ()
    steps: List[int] = []
    for frac in fractions:
        step = int(round(float(reference_step) * float(frac)))
        step = max(1, min(int(reference_step) - 1, step))
        steps.append(step)
    return tuple(sorted(set(steps)))


def _filter_summary(
    df: pd.DataFrame,
    *,
    variant_stages: Sequence[str],
    tracks: Sequence[str],
    seeds: Sequence[int],
    batches: Sequence[int],
) -> pd.DataFrame:
    work = df.copy()
    if variant_stages:
        work = work[work["variant_stage"].astype(str).isin(list(variant_stages))]
    if tracks:
        work = work[work["track"].astype(str).isin(list(tracks))]
    if seeds:
        work = work[pd.to_numeric(work["seed"], errors="coerce").isin(list(seeds))]
    if batches:
        work = work[pd.to_numeric(work["B"], errors="coerce").isin(list(batches))]
    return work.sort_values(["track", "variant_index", "seed", "B"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a filtered subset of selected variant runs into a fresh ablation directory.")
    parser.add_argument("--source-ablation-dir", type=str, required=True)
    parser.add_argument("--source-summary-csv", type=str, default="")
    parser.add_argument("--out-ablation-dir", type=str, required=True)
    parser.add_argument("--variant-stages", type=str, default="baseline,rope,muon,untie_embed")
    parser.add_argument("--tracks", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--B-list", type=str, default="")
    parser.add_argument("--checkpoint-fractions", type=str, default="")
    parser.add_argument("--num-target-loss", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    source_ablation_dir = Path(args.source_ablation_dir)
    source_summary_csv = Path(args.source_summary_csv) if args.source_summary_csv else (source_ablation_dir / "variant_concat_ablation_summary.csv")
    out_ablation_dir = Path(args.out_ablation_dir)
    out_root = out_ablation_dir.parent
    ablation_name = out_ablation_dir.name

    if out_ablation_dir.exists() and any(out_ablation_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing to reuse existing populated output directory without --force: {out_ablation_dir}")

    variant_stages = parse_list(args.variant_stages)
    track_filter = parse_list(args.tracks)
    seed_filter = parse_int_list(args.seeds) if str(args.seeds).strip() else []
    batch_filter = parse_int_list(args.B_list) if str(args.B_list).strip() else []
    checkpoint_fractions = parse_float_list(args.checkpoint_fractions) if str(args.checkpoint_fractions).strip() else []
    for frac in checkpoint_fractions:
        if not (0.0 < float(frac) < 1.0):
            raise SystemExit(f"--checkpoint-fractions entries must lie strictly between 0 and 1, got: {frac}")

    source_df = pd.read_csv(source_summary_csv)
    subset_df = _filter_summary(
        source_df,
        variant_stages=variant_stages,
        tracks=track_filter,
        seeds=seed_filter,
        batches=batch_filter,
    )
    if subset_df.empty:
        raise SystemExit("Filtered subset is empty.")

    stages = (
        subset_df[["variant_index", "variant_stage", "variant_combo"]]
        .drop_duplicates()
        .sort_values("variant_index")
        .itertuples(index=False, name=None)
    )
    stages = [(int(idx), str(stage), str(combo)) for idx, stage, combo in stages]
    tracks = sorted(subset_df["track"].astype(str).unique().tolist())
    seeds = sorted(pd.to_numeric(subset_df["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    batches = sorted(pd.to_numeric(subset_df["B"], errors="coerce").dropna().astype(int).unique().tolist())

    replay_rows: List[dict] = []
    replay_manifest_rows: List[dict] = []
    registry: Dict[Tuple[str, str, int, int], dict] = {}

    for _, row in subset_df.iterrows():
        source_run_dir = _resolve_run_dir(row, source_ablation_dir)
        if source_run_dir is None:
            raise FileNotFoundError(f"Could not resolve source run directory for {row.get('run_name', '')}")
        cfg = _load_run_config(source_run_dir)
        replay_cfg = replace(cfg, output_root=out_root)
        replay_checkpoint_steps: Tuple[int, ...] = ()
        if checkpoint_fractions:
            source_final_step = pd.to_numeric(row.get("final_step"), errors="coerce")
            if pd.isna(source_final_step):
                raise ValueError(f"Missing final_step for {row.get('run_name', '')}; cannot derive checkpoint schedule")
            replay_checkpoint_steps = _checkpoint_steps_from_reference(
                reference_step=int(source_final_step),
                fractions=checkpoint_fractions,
            )
            replay_cfg = replace(
                replay_cfg,
                save_checkpoints=True,
                checkpoint_every=0,
                checkpoint_steps=replay_checkpoint_steps,
            )
            print(
                "[replay] "
                f"variant_stage={row['variant_stage']} "
                f"track={row['track']} "
                f"B={int(row['B'])} "
                f"source_final_step={int(source_final_step)} "
                f"checkpoint_steps={','.join(str(x) for x in replay_checkpoint_steps)}",
                flush=True,
            )
        state = train_toy_run(
            config=replay_cfg,
            ablation_name=ablation_name,
            write_summary=True,
            save_spectra=True,
            device=args.device,
        )
        replay_rows.append(
            build_variant_concat_summary_row(
                state=state,
                task=str(row.get("task", replay_cfg.task)),
                track=str(row["track"]),
                variant_stage=str(row["variant_stage"]),
                variant_index=int(row["variant_index"]),
                variant_combo=str(row["variant_combo"]),
            )
        )
        registry[(str(row["track"]), str(row["variant_combo"]), int(row["seed"]), int(row["B"]))] = {
            "metrics_df": state.metrics_df,
            "run_dir": state.run_dir,
            "variant_stage": str(row["variant_stage"]),
            "variant_index": int(row["variant_index"]),
        }
        replay_manifest_rows.append(
            {
                **row.to_dict(),
                "source_run_dir": str(source_run_dir),
                "replayed_run_dir": str(state.run_dir),
                "replay_checkpoint_fractions": ",".join(f"{float(x):g}" for x in checkpoint_fractions),
                "replay_checkpoint_steps": ",".join(str(x) for x in replay_checkpoint_steps),
            }
        )

    replay_summary_df = pd.DataFrame(replay_rows).sort_values(["track", "variant_index", "seed", "B"]).reset_index(drop=True)
    matched_df, agg_df = build_variant_concat_matched_tables(
        tracks=tracks,
        seeds=seeds,
        batches=batches,
        stages=stages,
        registry=registry,
        num_target_loss=int(args.num_target_loss),
    )

    out_ablation_dir.mkdir(parents=True, exist_ok=True)
    replay_summary_df.to_csv(out_ablation_dir / "variant_concat_ablation_summary.csv", index=False)
    matched_df.to_csv(out_ablation_dir / "variant_concat_ablation_matched_pairs.csv", index=False)
    agg_df.to_csv(out_ablation_dir / "variant_concat_ablation_matched_aggregate.csv", index=False)
    pd.DataFrame(replay_manifest_rows).to_csv(out_ablation_dir / "variant_concat_ablation_replay_manifest.csv", index=False)

    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_summary.csv'}")
    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_matched_pairs.csv'}")
    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_matched_aggregate.csv'}")
    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_replay_manifest.csv'}")


if __name__ == "__main__":
    main()
