from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    from .config import MeasurementConfig, RunConfig
    from .run_concat_batch_regime_ablation import (
        build_trial_row_from_state,
        build_variant_concat_matched_tables,
        build_variant_concat_summary_row,
        common_loss_targets,
        early_metric_row,
        js_divergence,
        load_cov_spectrum,
        parse_float_list,
        parse_int_list,
        parse_list,
        response_rows_from_selected,
        safe_spearman,
        selection_key,
    )
    from .runner import matched_loss_rows, train_toy_run
except ImportError:
    from config import MeasurementConfig, RunConfig
    from run_concat_batch_regime_ablation import (
        build_trial_row_from_state,
        build_variant_concat_matched_tables,
        build_variant_concat_summary_row,
        common_loss_targets,
        early_metric_row,
        js_divergence,
        load_cov_spectrum,
        parse_float_list,
        parse_int_list,
        parse_list,
        response_rows_from_selected,
        safe_spearman,
        selection_key,
    )
    from runner import matched_loss_rows, train_toy_run


class _SimpleProgress:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.unit = str(unit)
        self.current = 0
        self.postfix = ""
        self._last_render = 0.0
        self._closed = False
        self._render(force=True)

    def _line(self) -> str:
        if self.total <= 0:
            base = f"{self.desc}: {self.current} {self.unit}"
        else:
            frac = min(max(float(self.current) / float(self.total), 0.0), 1.0)
            width = 28
            filled = int(round(width * frac))
            bar = "#" * filled + "-" * (width - filled)
            base = f"{self.desc} [{bar}] {self.current}/{self.total} ({100.0 * frac:5.1f}%)"
        if self.postfix:
            base = f"{base} | {self.postfix}"
        return base

    def _render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_render) < 0.05:
            return
        print(f"\r{self._line()}", end="", flush=True)
        self._last_render = now

    def update(self, n: int = 1, postfix: str = "") -> None:
        self.current += int(n)
        if postfix:
            self.postfix = str(postfix)
        self._render()

    def write(self, message: str) -> None:
        if self._closed:
            print(message, flush=True)
            return
        print("\r" + " " * max(len(self._line()), 1), end="\r", flush=True)
        print(message, flush=True)
        self._render(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._render(force=True)
        print(flush=True)


class _TqdmProgress:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self._bar = _tqdm(total=max(0, int(total)), desc=str(desc), unit=str(unit), dynamic_ncols=True)

    def update(self, n: int = 1, postfix: str = "") -> None:
        if postfix:
            self._bar.set_postfix_str(str(postfix), refresh=False)
        self._bar.update(int(n))

    def write(self, message: str) -> None:
        self._bar.write(str(message))

    def close(self) -> None:
        self._bar.close()


def _make_progress(total: int, desc: str, unit: str):
    if _tqdm is not None:
        return _TqdmProgress(total=total, desc=desc, unit=unit)
    return _SimpleProgress(total=total, desc=desc, unit=unit)


def _log(message: str, progress=None) -> None:
    if progress is not None:
        progress.write(str(message))
    else:
        print(str(message), flush=True)


def _format_batch_value_map(values: Dict[int, int]) -> str:
    if not values:
        return "<none>"
    return ", ".join(f"{int(batch)}:{int(value)}" for batch, value in sorted(values.items()))


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


def _load_metrics_df(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics_over_time.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_run_summary_dict(run_dir: Path) -> dict:
    csv_path = run_dir / "run_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            return df.iloc[0].to_dict()
    json_path = run_dir / "run_summary.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


def _resolve_metric_name(target_loss_metric: str) -> str:
    return "loss" if str(target_loss_metric).strip().lower() == "val" else "train_loss"


def _analysis_snapshot(
    metrics_df: pd.DataFrame,
    *,
    metric_name: str,
    target_loss: float | None,
    min_steps: int,
) -> Tuple[pd.Series, bool]:
    if metrics_df.empty:
        raise ValueError("metrics_df is empty")
    work = metrics_df.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    for col in (
        metric_name,
        "loss",
        "train_loss",
        "rankme",
        "alpha_head",
        "alpha_tail",
        "grad_rankme",
        "grad_alpha_head",
        "grad_alpha_tail",
        "tokens_seen",
        "train_time_s",
        "val_time_s",
        "checkpoint_time_s",
        "measurement_time_s",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    if work.empty:
        raise ValueError("metrics_df has no valid steps")
    if target_loss is None or not np.isfinite(float(target_loss)):
        return work.iloc[-1], False
    eligible = work
    if int(min_steps) > 0:
        eligible = eligible[eligible["step"] >= int(min_steps)]
    if metric_name in eligible.columns:
        eligible = eligible[pd.to_numeric(eligible[metric_name], errors="coerce") <= float(target_loss)]
        if not eligible.empty:
            return eligible.iloc[0], True
    return work.iloc[-1], False


def _retarget_trial_row(
    row: dict,
    *,
    target_loss: float | None,
    target_loss_metric: str,
    target_loss_min_steps: int,
) -> dict:
    run_dir = Path(str(row["run_dir"]))
    metrics_df = _load_metrics_df(run_dir)
    metric_name = _resolve_metric_name(target_loss_metric)
    chosen, hit_target = _analysis_snapshot(
        metrics_df,
        metric_name=metric_name,
        target_loss=target_loss,
        min_steps=target_loss_min_steps,
    )
    out = dict(row)
    out["source_target_loss"] = row.get("target_loss", np.nan)
    out["source_hit_target"] = row.get("hit_target", False)
    out["source_stop_reason"] = row.get("stop_reason", "")
    out["source_final_step"] = row.get("final_step", np.nan)
    out["source_tokens_seen"] = row.get("tokens_seen", np.nan)
    out["target_loss"] = float(target_loss) if target_loss is not None else np.nan
    out["target_loss_metric"] = str(target_loss_metric)
    out["hit_target"] = bool(hit_target)
    out["final_step"] = int(chosen.get("step", out.get("final_step", 0)))
    out["final_val_loss"] = float(chosen.get("loss", out.get("final_val_loss", np.nan)))
    out["final_train_loss"] = float(chosen.get("train_loss", out.get("final_train_loss", np.nan)))
    out["rankme"] = float(chosen.get("rankme", out.get("rankme", np.nan)))
    out["alpha_head"] = float(chosen.get("alpha_head", out.get("alpha_head", np.nan)))
    out["alpha_tail"] = float(chosen.get("alpha_tail", out.get("alpha_tail", np.nan)))
    out["grad_rankme"] = float(chosen.get("grad_rankme", out.get("grad_rankme", np.nan)))
    out["grad_alpha_head"] = float(chosen.get("grad_alpha_head", out.get("grad_alpha_head", np.nan)))
    out["grad_alpha_tail"] = float(chosen.get("grad_alpha_tail", out.get("grad_alpha_tail", np.nan)))
    out["tokens_seen"] = float(chosen.get("tokens_seen", out.get("tokens_seen", np.nan)))
    out["train_time_s"] = float(chosen.get("train_time_s", out.get("train_time_s", np.nan)))
    out["val_time_s"] = float(chosen.get("val_time_s", out.get("val_time_s", np.nan)))
    out["checkpoint_time_s"] = float(chosen.get("checkpoint_time_s", out.get("checkpoint_time_s", np.nan)))
    out["measurement_time_s"] = float(chosen.get("measurement_time_s", out.get("measurement_time_s", np.nan)))
    out["stop_reason"] = "target_loss" if hit_target else str(row.get("stop_reason", "max_steps"))
    out["stopped_early"] = bool(hit_target or row.get("stopped_early", False))
    out["selection_source"] = "source_retarget"
    return out


def _nearest_template_row(
    df: pd.DataFrame,
    *,
    track: str,
    variant_combo: str,
    seed: int,
    batch: int,
) -> pd.Series:
    sub = df[
        (df["track"].astype(str) == str(track))
        & (df["variant_combo"].astype(str) == str(variant_combo))
        & (pd.to_numeric(df["seed"], errors="coerce") == int(seed))
    ].copy()
    if sub.empty:
        raise KeyError(f"No template row found for track={track} variant_combo={variant_combo} seed={seed}")
    sub["B_num"] = pd.to_numeric(sub["B"], errors="coerce")
    sub["dist"] = (sub["B_num"] - int(batch)).abs()
    sub = sub.sort_values(["dist", "B_num"])
    return sub.iloc[0]


def _interpolate_lr(batch: int, anchors: pd.DataFrame) -> float:
    work = anchors.copy()
    work["B_num"] = pd.to_numeric(work["B"], errors="coerce")
    work["lr_num"] = pd.to_numeric(work["lr"], errors="coerce")
    work = work.dropna(subset=["B_num", "lr_num"]).sort_values("B_num")
    if work.empty:
        raise ValueError("Cannot interpolate LR without anchors")
    exact = work[np.isclose(work["B_num"], float(batch))]
    if not exact.empty:
        return float(exact.iloc[0]["lr_num"])
    left = work[work["B_num"] < int(batch)]
    right = work[work["B_num"] > int(batch)]
    if left.empty and right.empty:
        raise ValueError("Cannot interpolate LR with zero anchors")
    if left.empty:
        return float(right.iloc[0]["lr_num"])
    if right.empty:
        return float(left.iloc[-1]["lr_num"])
    left_row = left.iloc[-1]
    right_row = right.iloc[0]
    log_b0 = np.log(float(left_row["B_num"]))
    log_b1 = np.log(float(right_row["B_num"]))
    log_lr0 = np.log(float(left_row["lr_num"]))
    log_lr1 = np.log(float(right_row["lr_num"]))
    frac = (np.log(float(batch)) - log_b0) / max(log_b1 - log_b0, 1e-12)
    return float(np.exp((1.0 - frac) * log_lr0 + frac * log_lr1))


def _build_replay_manifest_row(
    selected_row: dict,
    *,
    selected_run_dir: str,
    reuse_mode: str,
) -> dict:
    out = dict(selected_row)
    out["selected_run_dir"] = str(selected_run_dir)
    out["selected_run_reuse_mode"] = str(reuse_mode)
    return out


def _parse_optional_filter(text: str) -> List[str]:
    return parse_list(text) if str(text).strip() else []


def _parse_optional_int_filter(text: str) -> List[int]:
    return parse_int_list(text) if str(text).strip() else []


def _parse_batch_value_map(text: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not str(text).strip():
        return out
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"Expected batch:value entry, got: {token}")
        batch_text, value_text = token.split(":", 1)
        batch = int(batch_text.strip())
        value = int(value_text.strip())
        if batch <= 0:
            raise SystemExit(f"Batch must be positive in batch:value entry: {token}")
        if value < 0:
            raise SystemExit(f"Value must be nonnegative in batch:value entry: {token}")
        out[int(batch)] = int(value)
    return out


def _batch_scale_factor(batch: int, reference_batch: Optional[int]) -> float:
    if reference_batch is None or int(reference_batch) <= 0:
        return 1.0
    return max(1.0, float(int(reference_batch)) / float(max(int(batch), 1)))


def _scaled_max_steps(base_steps: int, batch: int, reference_batch: Optional[int]) -> int:
    return max(1, int(np.ceil(float(base_steps) * _batch_scale_factor(batch, reference_batch))))


def _scaled_checkpoint_every(base_interval: int, batch: int, reference_batch: Optional[int]) -> int:
    if int(base_interval) <= 0:
        return 0
    return max(1, int(np.ceil(float(base_interval) * _batch_scale_factor(batch, reference_batch))))


def _effective_max_steps(
    base_steps: int,
    batch: int,
    reference_batch: Optional[int],
    batch_max_steps_overrides: Dict[int, int],
) -> int:
    if int(batch) in batch_max_steps_overrides:
        return max(1, int(batch_max_steps_overrides[int(batch)]))
    return _scaled_max_steps(base_steps, batch, reference_batch)


def _effective_checkpoint_every(
    base_interval: int,
    batch: int,
    reference_batch: Optional[int],
    batch_checkpoint_overrides: Dict[int, int],
) -> int:
    if int(batch) in batch_checkpoint_overrides:
        return max(0, int(batch_checkpoint_overrides[int(batch)]))
    return _scaled_checkpoint_every(base_interval, batch, reference_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget and extend an existing concat-batch-regime sweep with local LR scans.")
    parser.add_argument("--source-ablation-dir", type=str, required=True)
    parser.add_argument("--source-selected-ablation-dir", type=str, default="")
    parser.add_argument("--tracks", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--variant-stages", type=str, default="")
    parser.add_argument("--all-batches", type=str, default="32,64,128,256,512")
    parser.add_argument("--scan-batches", type=str, default="64,256")
    parser.add_argument("--target-loss", type=float, default=2.5)
    parser.add_argument("--target-loss-metric", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--target-loss-patience", type=int, default=1)
    parser.add_argument("--target-loss-min-steps", type=int, default=0)
    parser.add_argument("--num-target-loss", type=int, default=5)
    parser.add_argument("--early-checkpoints", type=str, default="200,400,800")
    parser.add_argument("--lr-multipliers", type=str, default="0.5,0.75,1.0,1.4,2.0")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--batch-max-steps",
        type=str,
        default="",
        help="Optional comma-separated batch:max_steps overrides, e.g. 32:40000,64:20000,128:10000.",
    )
    parser.add_argument(
        "--batch-checkpoint-every",
        type=str,
        default="",
        help="Optional comma-separated batch:checkpoint_every overrides for selected replays, e.g. 32:1000,64:500,128:250.",
    )
    parser.add_argument(
        "--step-scale-reference-batch",
        type=int,
        default=0,
        help="If > 0, batches below this reference get proportionally more max_steps/checkpoint spacing (e.g. 128->64 doubles both).",
    )
    parser.add_argument("--replay-all-selected", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selected-ablation-name", type=str, default="variant_concat_ablation")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--selected-out-root", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    source_ablation_dir = Path(args.source_ablation_dir)
    source_selected_ablation_dir = (
        Path(args.source_selected_ablation_dir)
        if str(args.source_selected_ablation_dir).strip()
        else (source_ablation_dir.parent / "concat_batch_regime_selected_runs" / "variant_concat_ablation")
    )
    out_root = Path(args.out_root)
    selected_out_root = Path(args.selected_out_root) if str(args.selected_out_root).strip() else (out_root / "concat_batch_regime_selected_runs")
    out_dir = out_root / "concat_batch_regime_ablation"
    selected_out_dir = selected_out_root / args.selected_ablation_name

    if ((out_dir.exists() and any(out_dir.iterdir())) or (selected_out_dir.exists() and any(selected_out_dir.iterdir()))) and not args.force:
        raise SystemExit(f"Refusing to reuse populated output dirs without --force: {out_dir} {selected_out_dir}")

    trials_path = source_ablation_dir / "concat_batch_regime_trials.csv"
    selected_path = source_ablation_dir / "concat_batch_regime_selected_lrs.csv"
    replay_manifest_path = source_ablation_dir / "concat_batch_regime_selected_replay_manifest.csv"
    selected_summary_path = source_selected_ablation_dir / "variant_concat_ablation_summary.csv"

    source_trials_df = pd.read_csv(trials_path)
    source_selected_df = pd.read_csv(selected_path)
    source_replay_manifest_df = pd.read_csv(replay_manifest_path)
    source_selected_summary_df = pd.read_csv(selected_summary_path)

    track_filter = _parse_optional_filter(args.tracks)
    stage_filter = _parse_optional_filter(args.variant_stages)
    seed_filter = _parse_optional_int_filter(args.seeds)
    all_batches = parse_int_list(args.all_batches)
    step_scale_reference_batch = int(args.step_scale_reference_batch) if int(args.step_scale_reference_batch) > 0 else None
    batch_max_steps_overrides = _parse_batch_value_map(args.batch_max_steps)
    batch_checkpoint_overrides = _parse_batch_value_map(args.batch_checkpoint_every)
    early_checkpoints = parse_int_list(args.early_checkpoints)
    lr_multipliers = parse_float_list(args.lr_multipliers)
    target_metric_name = _resolve_metric_name(args.target_loss_metric)

    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if track_filter:
            work = work[work["track"].astype(str).isin(track_filter)]
        if stage_filter:
            work = work[work["variant_stage"].astype(str).isin(stage_filter)]
        if seed_filter:
            work = work[pd.to_numeric(work["seed"], errors="coerce").isin(seed_filter)]
        if all_batches:
            work = work[pd.to_numeric(work["B"], errors="coerce").isin(all_batches)]
        return work.reset_index(drop=True)

    source_trials_df = _apply_filters(source_trials_df)
    source_selected_df = _apply_filters(source_selected_df)
    source_replay_manifest_df = _apply_filters(source_replay_manifest_df)
    source_selected_summary_df = _apply_filters(source_selected_summary_df)

    if source_trials_df.empty:
        raise SystemExit("Filtered source trials are empty.")

    default_source_max_steps = 0
    if not source_selected_df.empty:
        default_source_max_steps = int(_load_run_config(Path(str(source_selected_df.iloc[0]["run_dir"]))).max_steps)
    elif not source_trials_df.empty:
        default_source_max_steps = int(_load_run_config(Path(str(source_trials_df.iloc[0]["run_dir"]))).max_steps)
    if default_source_max_steps <= 0:
        raise SystemExit("Could not determine source max_steps from the filtered source runs.")

    source_batch_max_steps: Dict[int, int] = {}
    if not source_selected_df.empty:
        grouped = source_selected_df.copy()
        grouped["B_num"] = pd.to_numeric(grouped["B"], errors="coerce")
        for batch, group in grouped.dropna(subset=["B_num"]).groupby("B_num"):
            sample_run_dir = Path(str(group.iloc[0]["run_dir"]))
            source_batch_max_steps[int(batch)] = int(_load_run_config(sample_run_dir).max_steps)

    scan_batch_set = set(parse_int_list(args.scan_batches) if str(args.scan_batches).strip() else [])
    for batch in all_batches:
        desired_max_steps = _effective_max_steps(
            default_source_max_steps,
            int(batch),
            step_scale_reference_batch,
            batch_max_steps_overrides,
        )
        source_max_steps = source_batch_max_steps.get(int(batch))
        if source_max_steps is None or int(source_max_steps) != int(desired_max_steps):
            scan_batch_set.add(int(batch))
    scan_batches = sorted(int(x) for x in scan_batch_set)
    reuse_batches = sorted(int(b) for b in all_batches if int(b) not in scan_batch_set)

    effective_max_steps_by_batch = {
        int(batch): _effective_max_steps(
            default_source_max_steps,
            int(batch),
            step_scale_reference_batch,
            batch_max_steps_overrides,
        )
        for batch in all_batches
    }
    effective_checkpoint_by_batch = {
        int(batch): _effective_checkpoint_every(
            int(args.checkpoint_every),
            int(batch),
            step_scale_reference_batch,
            batch_checkpoint_overrides,
        )
        for batch in all_batches
    }

    tracks = sorted(source_trials_df["track"].astype(str).unique().tolist())
    seeds = sorted(pd.to_numeric(source_trials_df["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    stages = [
        (int(r["variant_index"]), str(r["variant_stage"]), str(r["variant_combo"]))
        for _, r in (
            source_trials_df[["variant_index", "variant_stage", "variant_combo"]]
            .drop_duplicates()
            .sort_values("variant_index")
            .iterrows()
        )
    ]

    source_trial_rows: List[dict] = []
    search_registry: Dict[Tuple[str, str, int, int], dict] = {}
    search_trial_registry: Dict[Tuple[str, str, int, int, float], dict] = {}

    for row in source_trials_df.to_dict(orient="records"):
        retargeted = _retarget_trial_row(
            row,
            target_loss=args.target_loss,
            target_loss_metric=args.target_loss_metric,
            target_loss_min_steps=args.target_loss_min_steps,
        )
        source_trial_rows.append(retargeted)
        search_trial_registry[
            (
                str(retargeted["track"]),
                str(retargeted["variant_combo"]),
                int(retargeted["B"]),
                int(retargeted["seed"]),
                float(retargeted["lr"]),
            )
        ] = {
            "metrics_df": _load_metrics_df(Path(str(retargeted["run_dir"]))),
            "run_dir": Path(str(retargeted["run_dir"])),
            "trial_row": retargeted,
        }

    retargeted_source_trials_df = pd.DataFrame(source_trial_rows)
    retargeted_source_selected_rows: List[dict] = []
    for row in source_selected_df.to_dict(orient="records"):
        retargeted_source_selected_rows.append(
            _retarget_trial_row(
                row,
                target_loss=args.target_loss,
                target_loss_metric=args.target_loss_metric,
                target_loss_min_steps=args.target_loss_min_steps,
            )
        )
    retargeted_source_selected_df = pd.DataFrame(retargeted_source_selected_rows)

    if retargeted_source_selected_df.empty:
        anchor_group_count = 0
    else:
        anchor_group_count = len(
            retargeted_source_selected_df[
                ["track", "variant_index", "variant_stage", "variant_combo", "seed"]
            ]
            .drop_duplicates()
        )
    expected_search_trials = anchor_group_count * len(scan_batches) * len(lr_multipliers)

    _log(
        "Extension setup: "
        f"tracks={tracks} seeds={seeds} stages={[stage for _, stage, _ in stages]} "
        f"all_batches={all_batches} scan_batches={scan_batches} reuse_batches={reuse_batches}"
    )
    _log(
        "Batch schedule: "
        + ", ".join(
            f"B={batch}->steps={effective_max_steps_by_batch[int(batch)]},ckpt={effective_checkpoint_by_batch[int(batch)]}"
            for batch in all_batches
        )
    )
    _log(
        "Overrides: "
        f"batch_max_steps={_format_batch_value_map(batch_max_steps_overrides)} "
        f"batch_checkpoint_every={_format_batch_value_map(batch_checkpoint_overrides)} "
        f"step_scale_reference_batch={step_scale_reference_batch}"
    )
    _log(
        f"Loaded {len(source_trials_df)} source trials, {len(source_selected_df)} source selected rows, "
        f"expecting about {expected_search_trials} scanned trial runs."
    )

    selected_rows: List[dict] = []
    old_selected_keys: set[Tuple[str, str, int, int]] = set()

    if not retargeted_source_trials_df.empty:
        group_cols = ["task", "track", "variant_index", "variant_stage", "variant_combo", "B", "seed"]
        reuse_trial_df = retargeted_source_trials_df[
            pd.to_numeric(retargeted_source_trials_df["B"], errors="coerce").isin(reuse_batches)
        ].copy()
        for _, group in reuse_trial_df.groupby(group_cols):
            records = group.to_dict(orient="records")
            best = min(
                records,
                key=lambda row: selection_key(
                    row,
                    target_loss=args.target_loss,
                    target_metric_name=target_metric_name,
                ),
            )
            best = dict(best)
            best["selection_mode"] = "target_loss" if args.target_loss is not None else "best_final_val"
            best["selection_source"] = "source_retarget"
            selected_rows.append(best)
            key = (str(best["track"]), str(best["variant_combo"]), int(best["B"]), int(best["seed"]))
            old_selected_keys.add(key)
            search_registry[key] = search_trial_registry[
                (
                    str(best["track"]),
                    str(best["variant_combo"]),
                    int(best["B"]),
                    int(best["seed"]),
                    float(best["lr"]),
                )
            ]

    retargeted_old_selected_df = pd.DataFrame(selected_rows)

    template_df = source_replay_manifest_df.copy()
    template_df["selected_run_dir"] = template_df["selected_run_dir"].astype(str)

    new_trial_rows: List[dict] = []
    new_selected_rows: List[dict] = []
    search_progress = _make_progress(expected_search_trials, "Search trials", "trial")

    try:
        for track in tracks:
            for variant_index, variant_stage, variant_combo in stages:
                for seed in seeds:
                    anchors = retargeted_source_selected_df[
                        (retargeted_source_selected_df["track"].astype(str) == str(track))
                        & (retargeted_source_selected_df["variant_combo"].astype(str) == str(variant_combo))
                        & (pd.to_numeric(retargeted_source_selected_df["seed"], errors="coerce") == int(seed))
                    ].copy()
                    if anchors.empty:
                        continue
                    template_row = _nearest_template_row(
                        template_df,
                        track=track,
                        variant_combo=variant_combo,
                        seed=seed,
                        batch=int(anchors["B"].iloc[0]),
                    )
                    template_cfg = _load_run_config(Path(str(template_row["selected_run_dir"])))
                    for batch in scan_batches:
                        lr_center = _interpolate_lr(int(batch), anchors)
                        trial_candidates: List[dict] = []
                        max_steps = _effective_max_steps(
                            int(template_cfg.max_steps),
                            int(batch),
                            step_scale_reference_batch,
                            batch_max_steps_overrides,
                        )
                        _log(
                            f"scan start stage={variant_stage} seed={seed} B={batch} "
                            f"lr_center={lr_center:.6g} max_steps={max_steps} candidates={len(lr_multipliers)}",
                            search_progress,
                        )
                        for cand_idx, mult in enumerate(lr_multipliers):
                            lr = float(lr_center) * float(mult)
                            if lr <= 0:
                                continue
                            cfg = replace(
                                template_cfg,
                                B=int(batch),
                                lr=float(lr),
                                seed=int(seed),
                                max_steps=max_steps,
                                target_loss=float(args.target_loss) if args.target_loss is not None else None,
                                target_loss_metric=str(args.target_loss_metric),
                                target_loss_patience=int(args.target_loss_patience),
                                target_loss_min_steps=int(args.target_loss_min_steps),
                                save_checkpoints=False,
                                checkpoint_every=0,
                                output_root=out_root,
                            )
                            state = train_toy_run(
                                config=cfg,
                                ablation_name="concat_batch_regime_ablation",
                                write_summary=True,
                                save_spectra=True,
                                device=args.device,
                            )
                            trial_row = build_trial_row_from_state(
                                state=state,
                                task=str(template_row["task"]),
                                track=track,
                                variant_index=int(variant_index),
                                variant_stage=variant_stage,
                                variant_combo=variant_combo,
                                target_loss=args.target_loss,
                                target_metric_name=target_metric_name,
                                search_mode="local_grid",
                                search_status="complete",
                                search_budget_steps=int(max_steps),
                                search_budget_tokens=float(cfg.max_steps * int(batch) * int(cfg.seq_len)),
                                search_budget_frac=1.0,
                                search_candidate_index=cand_idx,
                                search_rank=cand_idx + 1,
                                search_promoted=True,
                            )
                            trial_row["lr_center"] = float(lr_center)
                            trial_row["selection_source"] = "new_local_scan"
                            new_trial_rows.append(trial_row)
                            trial_candidates.append(trial_row)
                            search_trial_registry[
                                (
                                    str(trial_row["track"]),
                                    str(trial_row["variant_combo"]),
                                    int(trial_row["B"]),
                                    int(trial_row["seed"]),
                                    float(trial_row["lr"]),
                                )
                            ] = {
                                "metrics_df": state.metrics_df,
                                "run_dir": state.run_dir,
                                "trial_row": trial_row,
                            }
                            search_progress.update(
                                1,
                                postfix=f"{variant_stage} B={batch} seed={seed} lr={lr:.4g}",
                            )
                            _log(
                                f"scan done stage={variant_stage} seed={seed} B={batch} "
                                f"candidate={cand_idx + 1}/{len(lr_multipliers)} lr={lr:.6g} "
                                f"step={int(trial_row['final_step'])} val={float(trial_row['final_val_loss']):.4f} "
                                f"hit_target={bool(trial_row['hit_target'])}",
                                search_progress,
                            )
                        if not trial_candidates:
                            continue
                        best = min(
                            trial_candidates,
                            key=lambda row: selection_key(
                                row,
                                target_loss=args.target_loss,
                                target_metric_name=target_metric_name,
                            ),
                        )
                        best = dict(best)
                        best["selection_mode"] = "target_loss" if args.target_loss is not None else "best_final_val"
                        best["selection_source"] = "new_local_scan"
                        new_selected_rows.append(best)
                        search_registry[(track, variant_combo, int(batch), int(seed))] = search_trial_registry[
                            (
                                str(best["track"]),
                                str(best["variant_combo"]),
                                int(best["B"]),
                                int(best["seed"]),
                                float(best["lr"]),
                            )
                        ]
                        _log(
                            f"scan selected stage={variant_stage} seed={seed} B={batch} "
                            f"best_lr={float(best['lr']):.6g} step={int(best['final_step'])} "
                            f"val={float(best['final_val_loss']):.4f} hit_target={bool(best['hit_target'])}",
                            search_progress,
                        )
    finally:
        search_progress.close()

    reused_source_trials_df = retargeted_source_trials_df[
        pd.to_numeric(retargeted_source_trials_df["B"], errors="coerce").isin(reuse_batches)
    ].copy()
    trials_df = pd.concat([reused_source_trials_df, pd.DataFrame(new_trial_rows)], ignore_index=True)
    selected_df = pd.concat(
        [retargeted_old_selected_df, pd.DataFrame(new_selected_rows)],
        ignore_index=True,
    ).sort_values(["track", "variant_index", "seed", "B"]).reset_index(drop=True)

    source_selected_lookup = {
        (
            str(row["track"]),
            str(row["variant_combo"]),
            int(row["B"]),
            int(row["seed"]),
        ): row
        for _, row in source_selected_df.iterrows()
    }
    source_replay_lookup = {
        (
            str(row["track"]),
            str(row["variant_combo"]),
            int(row["B"]),
            int(row["seed"]),
        ): row
        for _, row in source_replay_manifest_df.iterrows()
    }
    source_summary_lookup = {
        (
            str(row["track"]),
            str(row["variant_combo"]),
            int(row["B"]),
            int(row["seed"]),
        ): row
        for _, row in source_selected_summary_df.iterrows()
    }

    matched_row_list: List[dict] = []
    pair_row_list: List[dict] = []
    if not selected_df.empty:
        for (track, variant_combo, seed), group in selected_df.groupby(["track", "variant_combo", "seed"]):
            selected_states = [search_registry[(track, variant_combo, int(r["B"]), int(seed))] for _, r in group.iterrows()]
            dfs = [x["metrics_df"] for x in selected_states]
            targets = common_loss_targets(dfs, num_targets=args.num_target_loss)
            if len(targets) == 0:
                continue
            matched_tables: Dict[int, pd.DataFrame] = {}
            for _, row in group.iterrows():
                key = (track, variant_combo, int(row["B"]), int(seed))
                st = search_registry[key]
                matched = matched_loss_rows(st["metrics_df"], targets)
                if matched.empty:
                    continue
                matched = matched.copy()
                matched["task"] = row["task"]
                matched["track"] = track
                matched["variant_combo"] = variant_combo
                matched["variant_stage"] = row["variant_stage"]
                matched["variant_index"] = int(row["variant_index"])
                matched["seed"] = int(seed)
                matched["B"] = int(row["B"])
                matched["lr"] = float(row["lr"])
                matched["run_dir"] = str(st["run_dir"])
                matched["hit_target"] = bool(row["hit_target"])
                matched_row_list.extend(matched.to_dict(orient="records"))
                matched_tables[int(row["B"])] = matched
            for target in targets:
                batches_present = sorted(matched_tables.keys())
                for i, left_b in enumerate(batches_present):
                    for right_b in batches_present[i + 1 :]:
                        left = matched_tables[left_b]
                        right = matched_tables[right_b]
                        lrow = left[np.isclose(left["matched_loss_target"], target)]
                        rrow = right[np.isclose(right["matched_loss_target"], target)]
                        if lrow.empty or rrow.empty:
                            continue
                        litem = lrow.iloc[0]
                        ritem = rrow.iloc[0]
                        s1 = load_cov_spectrum(Path(str(litem["run_dir"])), int(litem["step"]))
                        s2 = load_cov_spectrum(Path(str(ritem["run_dir"])), int(ritem["step"]))
                        pair_row_list.append(
                            {
                                "task": litem["task"],
                                "track": track,
                                "variant_combo": variant_combo,
                                "variant_stage": litem["variant_stage"],
                                "variant_index": int(litem["variant_index"]),
                                "seed": int(seed),
                                "matched_loss_target": float(target),
                                "left_B": int(left_b),
                                "right_B": int(right_b),
                                "left_lr": float(litem["lr"]),
                                "right_lr": float(ritem["lr"]),
                                "left_step": int(litem["step"]),
                                "right_step": int(ritem["step"]),
                                "left_match_error": float(litem.get("matched_loss_error", np.nan)),
                                "right_match_error": float(ritem.get("matched_loss_error", np.nan)),
                                "js_div_cov": js_divergence(s1, s2),
                            }
                        )

    early_rows: List[dict] = []
    selected_with_response = response_rows_from_selected(selected_df, target_loss=args.target_loss)
    predictors = [
        "rankme",
        "alpha_head",
        "alpha_tail",
        "grad_rankme",
        "grad_alpha_head",
        "grad_alpha_tail",
        "act_grad_head_gap",
        "act_grad_tail_gap",
    ]
    if not selected_with_response.empty:
        for (track, variant_combo), group in selected_with_response.groupby(["track", "variant_combo"]):
            for checkpoint in early_checkpoints:
                point_rows: List[dict] = []
                for _, row in group.iterrows():
                    key = (track, variant_combo, int(row["B"]), int(row["seed"]))
                    state = search_registry.get(key)
                    if state is None:
                        continue
                    early = early_metric_row(state["metrics_df"], checkpoint)
                    if early is None:
                        continue
                    item = {
                        "task": row["task"],
                        "track": track,
                        "variant_combo": variant_combo,
                        "variant_stage": row["variant_stage"],
                        "variant_index": int(row["variant_index"]),
                        "seed": int(row["seed"]),
                        "B": int(row["B"]),
                        "checkpoint": int(checkpoint),
                        "response_metric": row["response_metric"],
                        "response_value": float(row["response_value"]),
                        "rankme": float(early.get("rankme", np.nan)),
                        "alpha_head": float(early.get("alpha_head", np.nan)),
                        "alpha_tail": float(early.get("alpha_tail", np.nan)),
                        "grad_rankme": float(early.get("grad_rankme", np.nan)),
                        "grad_alpha_head": float(early.get("grad_alpha_head", np.nan)),
                        "grad_alpha_tail": float(early.get("grad_alpha_tail", np.nan)),
                    }
                    item["act_grad_head_gap"] = item["alpha_head"] - item["grad_alpha_head"]
                    item["act_grad_tail_gap"] = item["alpha_tail"] - item["grad_alpha_tail"]
                    point_rows.append(item)
                if not point_rows:
                    continue
                point_df = pd.DataFrame(point_rows)
                for predictor in predictors:
                    early_rows.append(
                        {
                            "task": point_df["task"].iloc[0],
                            "track": track,
                            "variant_combo": variant_combo,
                            "variant_stage": point_df["variant_stage"].iloc[0],
                            "variant_index": int(point_df["variant_index"].iloc[0]),
                            "checkpoint": int(checkpoint),
                            "predictor": predictor,
                            "response_metric": point_df["response_metric"].iloc[0],
                            "spearman_r": safe_spearman(
                                point_df[predictor].to_numpy(dtype=float),
                                point_df["response_value"].to_numpy(dtype=float),
                            ),
                            "n": int(len(point_df)),
                        }
                    )

    replay_rows: List[dict] = []
    replay_manifest_rows: List[dict] = []
    selected_replay_registry: Dict[Tuple[str, str, int, int], dict] = {}
    replay_progress = _make_progress(len(selected_df), "Selected replays", "run")

    try:
        for _, row in selected_df.iterrows():
            key = (str(row["track"]), str(row["variant_combo"]), int(row["B"]), int(row["seed"]))
            source_selected_row = source_selected_lookup.get(key)
            source_replay_row = source_replay_lookup.get(key)
            can_reuse = False
            desired_replay_max_steps = _effective_max_steps(
                default_source_max_steps,
                int(row["B"]),
                step_scale_reference_batch,
                batch_max_steps_overrides,
            )
            desired_replay_checkpoint_every = _effective_checkpoint_every(
                int(args.checkpoint_every),
                int(row["B"]),
                step_scale_reference_batch,
                batch_checkpoint_overrides,
            )
            if source_selected_row is not None and source_replay_row is not None:
                source_replay_cfg = _load_run_config(Path(str(source_replay_row["selected_run_dir"])))
                can_reuse = (
                    np.isclose(float(source_selected_row["lr"]), float(row["lr"]))
                    and str(source_selected_row["run_dir"]) == str(row["run_dir"])
                    and int(source_replay_cfg.max_steps) == int(desired_replay_max_steps)
                    and int(source_replay_cfg.checkpoint_every) == int(desired_replay_checkpoint_every)
                )
            should_replay = args.replay_all_selected or not can_reuse
            if should_replay:
                _log(
                    f"replay start stage={row['variant_stage']} seed={int(row['seed'])} B={int(row['B'])} "
                    f"lr={float(row['lr']):.6g} max_steps={desired_replay_max_steps} "
                    f"checkpoint_every={desired_replay_checkpoint_every}",
                    replay_progress,
                )
                template_row = _nearest_template_row(
                    template_df,
                    track=str(row["track"]),
                    variant_combo=str(row["variant_combo"]),
                    seed=int(row["seed"]),
                    batch=int(row["B"]),
                )
                template_cfg = _load_run_config(Path(str(template_row["selected_run_dir"])))
                replay_max_steps = _effective_max_steps(
                    int(template_cfg.max_steps),
                    int(row["B"]),
                    step_scale_reference_batch,
                    batch_max_steps_overrides,
                )
                replay_checkpoint_every = _effective_checkpoint_every(
                    int(args.checkpoint_every),
                    int(row["B"]),
                    step_scale_reference_batch,
                    batch_checkpoint_overrides,
                )
                cfg = replace(
                    template_cfg,
                    B=int(row["B"]),
                    lr=float(row["lr"]),
                    seed=int(row["seed"]),
                    max_steps=replay_max_steps,
                    target_loss=float(args.target_loss) if args.target_loss is not None else None,
                    target_loss_metric=str(args.target_loss_metric),
                    target_loss_patience=int(args.target_loss_patience),
                    target_loss_min_steps=int(args.target_loss_min_steps),
                    save_checkpoints=True,
                    checkpoint_every=int(replay_checkpoint_every),
                    output_root=selected_out_root,
                )
                state = train_toy_run(
                    config=cfg,
                    ablation_name=args.selected_ablation_name,
                    write_summary=True,
                    save_spectra=True,
                    device=args.device,
                )
                replay_rows.append(
                    build_variant_concat_summary_row(
                        state=state,
                        task=str(row["task"]),
                        track=str(row["track"]),
                        variant_stage=str(row["variant_stage"]),
                        variant_index=int(row["variant_index"]),
                        variant_combo=str(row["variant_combo"]),
                    )
                )
                selected_replay_registry[key] = {
                    "metrics_df": state.metrics_df,
                    "run_dir": state.run_dir,
                    "variant_stage": str(row["variant_stage"]),
                    "variant_index": int(row["variant_index"]),
                }
                replay_manifest_rows.append(
                    _build_replay_manifest_row(
                        row.to_dict(),
                        selected_run_dir=str(state.run_dir),
                        reuse_mode="replayed_for_extension",
                    )
                )
                replay_progress.update(
                    1,
                    postfix=f"{row['variant_stage']} B={int(row['B'])} seed={int(row['seed'])}",
                )
                _log(
                    f"replay done stage={row['variant_stage']} seed={int(row['seed'])} B={int(row['B'])} "
                    f"step={int(state.summary_row.get('final_step', 0))} stop_reason={state.summary_row.get('stop_reason', '')} "
                    f"run_dir={state.run_dir}",
                    replay_progress,
                )
            else:
                summary_row = source_summary_lookup[key]
                replay_rows.append(dict(summary_row))
                selected_replay_registry[key] = {
                    "metrics_df": _load_metrics_df(Path(str(summary_row["run_dir"]))),
                    "run_dir": Path(str(summary_row["run_dir"])),
                    "variant_stage": str(summary_row["variant_stage"]),
                    "variant_index": int(summary_row["variant_index"]),
                }
                replay_manifest_rows.append(
                    _build_replay_manifest_row(
                        row.to_dict(),
                        selected_run_dir=str(source_replay_row["selected_run_dir"]),
                        reuse_mode="reused_source_selected_run",
                    )
                )
                replay_progress.update(
                    1,
                    postfix=f"{row['variant_stage']} B={int(row['B'])} seed={int(row['seed'])}",
                )
                _log(
                    f"replay reused stage={row['variant_stage']} seed={int(row['seed'])} B={int(row['B'])} "
                    f"selected_run_dir={source_replay_row['selected_run_dir']}",
                    replay_progress,
                )
    finally:
        replay_progress.close()

    replay_summary_df = pd.DataFrame(replay_rows).sort_values(["track", "variant_index", "seed", "B"]).reset_index(drop=True)
    replay_matched_df, replay_agg_df = build_variant_concat_matched_tables(
        tracks=tracks,
        seeds=seeds,
        batches=all_batches,
        stages=stages,
        registry=selected_replay_registry,
        num_target_loss=args.num_target_loss,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_out_dir.mkdir(parents=True, exist_ok=True)

    trials_df.to_csv(out_dir / "concat_batch_regime_trials.csv", index=False)
    selected_df.to_csv(out_dir / "concat_batch_regime_selected_lrs.csv", index=False)
    selected_df.to_csv(out_dir / "constant_loss_stage_summary.csv", index=False)
    pd.DataFrame(matched_row_list).to_csv(out_dir / "concat_batch_regime_matched_loss_rows.csv", index=False)
    pd.DataFrame(pair_row_list).to_csv(out_dir / "concat_batch_regime_matched_pairs.csv", index=False)
    pd.DataFrame(early_rows).to_csv(out_dir / "concat_batch_regime_early_prediction.csv", index=False)
    pd.DataFrame(replay_manifest_rows).to_csv(out_dir / "concat_batch_regime_selected_replay_manifest.csv", index=False)

    replay_summary_df.to_csv(selected_out_dir / "variant_concat_ablation_summary.csv", index=False)
    replay_matched_df.to_csv(selected_out_dir / "variant_concat_ablation_matched_pairs.csv", index=False)
    replay_agg_df.to_csv(selected_out_dir / "variant_concat_ablation_matched_aggregate.csv", index=False)

    metadata = {
        "source_ablation_dir": str(source_ablation_dir),
        "source_selected_ablation_dir": str(source_selected_ablation_dir),
        "target_loss": float(args.target_loss) if args.target_loss is not None else None,
        "target_loss_metric": str(args.target_loss_metric),
        "all_batches": [int(x) for x in all_batches],
        "scan_batches": [int(x) for x in scan_batches],
        "reuse_batches": [int(x) for x in reuse_batches],
        "batch_max_steps_overrides": {str(k): int(v) for k, v in sorted(batch_max_steps_overrides.items())},
        "batch_checkpoint_overrides": {str(k): int(v) for k, v in sorted(batch_checkpoint_overrides.items())},
        "step_scale_reference_batch": int(step_scale_reference_batch) if step_scale_reference_batch is not None else None,
        "tracks": tracks,
        "seeds": seeds,
        "variant_stages": [stage for _, stage, _ in stages],
        "replay_all_selected": bool(args.replay_all_selected),
        "note": "Reused selected runs retain their original run_config target_loss; pass --target-loss explicitly to downstream probe scripts when targeting 2.5. Batch-specific max_steps/checkpoint overrides take precedence over step_scale_reference_batch when provided.",
    }
    (out_dir / "extension_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote: {out_dir / 'concat_batch_regime_trials.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_selected_lrs.csv'}")
    print(f"Wrote: {out_dir / 'constant_loss_stage_summary.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_matched_loss_rows.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_matched_pairs.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_early_prediction.csv'}")
    print(f"Wrote: {out_dir / 'concat_batch_regime_selected_replay_manifest.csv'}")
    print(f"Wrote: {out_dir / 'extension_metadata.json'}")
    print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_summary.csv'}")
    print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_matched_pairs.csv'}")
    print(f"Wrote: {selected_out_dir / 'variant_concat_ablation_matched_aggregate.csv'}")


if __name__ == "__main__":
    main()
