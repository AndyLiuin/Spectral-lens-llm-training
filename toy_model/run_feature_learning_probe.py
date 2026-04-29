from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from .feature_learning_utils import (
        ProbeSplit,
        build_fixed_band_probe_split,
        build_matched_band_probe_split,
        run_feature_probe_once,
    )
    from .models_modarith import build_modarith_model
except ImportError:
    from feature_learning_utils import (
        ProbeSplit,
        build_fixed_band_probe_split,
        build_matched_band_probe_split,
        run_feature_probe_once,
    )
    from models_modarith import build_modarith_model


def parse_bands(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"Band must be c0:o0, got: {s}")
        c0_s, o0_s = s.split(":", 1)
        out.append((int(c0_s.strip()), int(o0_s.strip())))
    return out


def parse_steps(text: str) -> Optional[List[int]]:
    s = str(text).strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_fracs(text: str) -> Optional[List[float]]:
    s = str(text).strip()
    if not s:
        return None
    out: List[float] = []
    for part in s.split(","):
        val = float(part.strip())
        if not np.isfinite(val):
            continue
        out.append(min(max(val, 0.0), 1.0))
    return out


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_extension_target_defaults(ablation_dir: Path) -> Tuple[Optional[float], str]:
    candidate = ablation_dir.parent.parent / "concat_batch_regime_ablation" / "extension_metadata.json"
    if not candidate.exists():
        return None, ""
    try:
        obj = json.loads(candidate.read_text())
    except Exception:
        return None, ""

    target_loss_raw = obj.get("target_loss", None)
    target_metric = str(obj.get("target_loss_metric", "")).strip().lower()
    try:
        target_loss = float(target_loss_raw) if target_loss_raw is not None else None
    except Exception:
        target_loss = None
    return target_loss, target_metric


def _resolve_run_dir(row: pd.Series, ablation_dir: Path) -> Optional[Path]:
    candidates = []
    if "run_name" in row and isinstance(row["run_name"], str):
        candidates.append(ablation_dir / row["run_name"])
    if "run_dir" in row and isinstance(row["run_dir"], str):
        p = Path(row["run_dir"])
        candidates.append(p if p.is_absolute() else (ablation_dir.parent.parent / p))
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_checkpoint_steps(ckpt_dir: Path) -> List[int]:
    steps = []
    for p in sorted(ckpt_dir.glob("model_step_*.pt")):
        name = p.stem
        try:
            steps.append(int(name.split("_")[-1]))
        except Exception:
            continue
    return sorted(set(steps))


def _checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"model_step_{int(step):06d}.pt"


def _load_metrics_df(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics_over_time.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_run_summary(run_dir: Path) -> dict:
    path = run_dir / "run_summary.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _nearest_checkpoint_step(available_steps: List[int], desired_step: int) -> Optional[int]:
    if not available_steps:
        return None
    return min(available_steps, key=lambda step: (abs(int(step) - int(desired_step)), int(step)))


def _resolve_target_metric_name(target_metric: str) -> str:
    return "loss" if str(target_metric).strip().lower() == "val" else "train_loss"


def _resolve_target_checkpoint_step(
    *,
    metrics_df: pd.DataFrame,
    available_steps: List[int],
    target_loss: Optional[float],
    target_metric: str,
    min_steps: int,
) -> Optional[int]:
    if target_loss is None or not np.isfinite(float(target_loss)) or metrics_df.empty or not available_steps:
        return None
    metric_name = _resolve_target_metric_name(target_metric)
    if metric_name not in metrics_df.columns or "step" not in metrics_df.columns:
        return None
    work = metrics_df.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    work[metric_name] = pd.to_numeric(work[metric_name], errors="coerce")
    work = work.dropna(subset=["step", metric_name])
    if min_steps > 0:
        work = work[work["step"] >= int(min_steps)]
    if work.empty:
        return None
    idx = (work[metric_name] - float(target_loss)).abs().idxmin()
    desired_step = int(work.loc[idx, "step"])
    return _nearest_checkpoint_step(available_steps, desired_step)


def _merge_step_source(step_sources: Dict[int, Set[str]], step: Optional[int], source: str) -> None:
    if step is None:
        return
    step_sources.setdefault(int(step), set()).add(source)


def _primary_alignment_source(step_sources: Dict[int, Set[str]], step: int) -> Tuple[str, float]:
    sources = sorted(step_sources.get(int(step), set()))
    for src in sources:
        if src.startswith("frac:"):
            try:
                return src, float(src.split(":", 1)[1])
            except Exception:
                continue
    return (sources[0] if sources else "step", np.nan)


def _load_model_from_run_config(run_cfg: dict, device: str) -> torch.nn.Module:
    if str(run_cfg.get("task", "")).strip().lower() != "mod_arith_lm":
        raise ValueError("Feature-learning probe currently supports task=mod_arith_lm only.")
    model = build_modarith_model(
        track=run_cfg.get("track", "a"),
        vocab_size=int(run_cfg.get("vocab_size", 1024)),
        seq_len=int(run_cfg.get("seq_len", 64)),
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("num_layers", 2)),
        ff_mult=int(run_cfg.get("ff_mult", 2)),
        variant=str(run_cfg.get("variant", "baseline")),
        window_size=int(run_cfg.get("window_size", 0) or 0),
        attention_scale=run_cfg.get("attention_scale", None),
        lm_head_softcap=float(run_cfg.get("lm_head_softcap", 30.0)),
    )
    return model.to(device)


def _is_mixed_or_noisy(run_cfg: dict) -> bool:
    return (
        int(run_cfg.get("mix_components_max", 1)) > 1
        or int(run_cfg.get("mix_components_min", 1)) > 1
        or float(run_cfg.get("component_weight_pareto_alpha", 0.0)) > 0.0
        or float(run_cfg.get("token_noise_std", 0.0)) > 0.0
    )


def _resolve_probe_types(requested: str, run_cfg: dict) -> List[str]:
    mode = str(requested).strip().lower()
    if mode == "auto":
        return ["clean_band", "matched_band"] if _is_mixed_or_noisy(run_cfg) else ["clean_band"]
    if mode == "clean":
        return ["clean_band"]
    if mode == "matched":
        return ["matched_band"]
    if mode == "both":
        return ["clean_band", "matched_band"]
    raise ValueError(f"Unknown probe_regime: {requested}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-learning FFT/causal/PCA probes over saved checkpoints.")
    parser.add_argument("--ablation-dir", type=str, default="toy_model/variant_concat_ablation")
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--bands", type=str, default="97:200,179:50")
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--checkpoint-fracs", type=str, default="")
    parser.add_argument("--include-initial-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-final-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-target-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--target-loss-metric", type=str, default="", choices=["", "val", "train"])
    parser.add_argument("--probe-size", type=int, default=12000)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--pca-k", type=int, default=8)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-regime", type=str, default="both", choices=["clean", "matched", "both", "auto"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (ablation_dir / "variant_concat_ablation_summary.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (ablation_dir / "feature_learning_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    bands = parse_bands(args.bands)
    checkpoint_filter = parse_steps(args.checkpoints)
    checkpoint_frac_filter = parse_fracs(args.checkpoint_fracs)
    extension_target_loss, extension_target_metric = _load_extension_target_defaults(ablation_dir)

    summary_df = pd.read_csv(summary_csv).sort_values(["track", "variant_index", "seed"])
    probe_cache: Dict[Tuple[object, ...], ProbeSplit] = {}

    summary_rows: List[dict] = []
    causal_rows: List[dict] = []
    pca_rows: List[dict] = []
    missing_runs: List[dict] = []
    checkpoint_manifest_rows: List[dict] = []

    for _, row in summary_df.iterrows():
        run_dir = _resolve_run_dir(row=row, ablation_dir=ablation_dir)
        if run_dir is None:
            missing_runs.append(
                {
                    "run_name": row.get("run_name", ""),
                    "reason": "run_dir_not_found",
                }
            )
            continue

        run_cfg_path = run_dir / "run_config.json"
        ckpt_dir = run_dir / "checkpoints"
        if not run_cfg_path.exists():
            missing_runs.append({"run_name": row.get("run_name", ""), "reason": "missing_run_config"})
            continue
        if not ckpt_dir.exists():
            missing_runs.append({"run_name": row.get("run_name", ""), "reason": "missing_checkpoints"})
            continue

        run_cfg = json.loads(run_cfg_path.read_text())
        run_summary = _load_run_summary(run_dir)
        metrics_df = _load_metrics_df(run_dir)
        all_ckpt_steps = _load_checkpoint_steps(ckpt_dir)
        if not all_ckpt_steps:
            missing_runs.append({"run_name": row.get("run_name", ""), "reason": "no_checkpoint_files"})
            continue
        available_set = set(all_ckpt_steps)
        step_sources: Dict[int, Set[str]] = {}
        final_step = run_summary.get("final_step", row.get("final_step", np.nan))
        final_step = int(final_step) if pd.notna(final_step) else int(all_ckpt_steps[-1])
        final_ckpt_step = _nearest_checkpoint_step(all_ckpt_steps, final_step)
        final_ckpt_step = int(final_ckpt_step) if final_ckpt_step is not None else int(all_ckpt_steps[-1])
        if checkpoint_filter is not None:
            for step in checkpoint_filter:
                if int(step) in available_set:
                    _merge_step_source(step_sources, int(step), "manual")
        if checkpoint_frac_filter is not None:
            for frac in checkpoint_frac_filter:
                desired_step = int(round(float(frac) * float(final_ckpt_step)))
                _merge_step_source(step_sources, _nearest_checkpoint_step(all_ckpt_steps, desired_step), f"frac:{frac:.6f}")
        if args.include_initial_checkpoint and all_ckpt_steps:
            _merge_step_source(step_sources, all_ckpt_steps[0], "initial")
        if args.include_final_checkpoint:
            _merge_step_source(step_sources, final_ckpt_step, "final")

        row_target_loss = row.get("target_loss", np.nan)
        cfg_target_loss = run_cfg.get("target_loss", np.nan)
        if args.target_loss is not None:
            target_loss = float(args.target_loss)
            target_loss_source = "cli"
        elif extension_target_loss is not None:
            target_loss = float(extension_target_loss)
            target_loss_source = "extension_metadata"
        elif pd.notna(cfg_target_loss):
            target_loss = float(cfg_target_loss)
            target_loss_source = "run_config"
        elif pd.notna(row_target_loss):
            target_loss = float(row_target_loss)
            target_loss_source = "summary_row"
        else:
            target_loss = None
            target_loss_source = "missing"

        if args.target_loss_metric:
            target_metric = str(args.target_loss_metric).strip().lower()
            target_metric_source = "cli"
        elif extension_target_metric:
            target_metric = extension_target_metric
            target_metric_source = "extension_metadata"
        else:
            cfg_target_metric = str(run_cfg.get("target_loss_metric", "")).strip().lower()
            row_target_metric = str(row.get("target_loss_metric", "")).strip().lower()
            if cfg_target_metric:
                target_metric = cfg_target_metric
                target_metric_source = "run_config"
            elif row_target_metric:
                target_metric = row_target_metric
                target_metric_source = "summary_row"
            else:
                target_metric = "val"
                target_metric_source = "default"

        if args.include_target_checkpoint:
            target_step = _resolve_target_checkpoint_step(
                metrics_df=metrics_df,
                available_steps=all_ckpt_steps,
                target_loss=target_loss,
                target_metric=target_metric,
                min_steps=int(run_cfg.get("target_loss_min_steps", row.get("target_loss_min_steps", 0)) or 0),
            )
            _merge_step_source(step_sources, target_step, "target")

        use_steps = sorted(step_sources.keys())
        if not use_steps:
            missing_runs.append({"run_name": row.get("run_name", ""), "reason": "no_probe_checkpoints_selected"})
            continue

        model = _load_model_from_run_config(run_cfg=run_cfg, device=device)

        for step in use_steps:
            ckpt_path = _checkpoint_path(ckpt_dir=ckpt_dir, step=step)
            obj = torch.load(ckpt_path, map_location=device)
            state = obj.get("model_state_dict", obj)
            model.load_state_dict(state, strict=True)
            model.eval()
            primary_align_key, primary_align_frac = _primary_alignment_source(step_sources, int(step))
            step_progress = (float(step) / float(final_ckpt_step)) if final_ckpt_step > 0 else np.nan
            step_progress_pct = int(round(100.0 * step_progress)) if np.isfinite(step_progress) else -1

            probe_types = _resolve_probe_types(args.probe_regime, run_cfg)

            for band_idx, (c0, o0) in enumerate(bands):
                for probe_type in probe_types:
                    key = (
                        probe_type,
                        int(c0),
                        int(o0),
                        int(run_cfg.get("seq_len", 64)),
                        int(run_cfg.get("vocab_size", 1024)),
                        float(run_cfg.get("min_step_frac", 0.125)),
                        bool(run_cfg.get("allow_noncoprime", True)),
                        float(run_cfg.get("noncoprime_prob", 0.3)),
                        int(run_cfg.get("mix_components_min", 1)),
                        int(run_cfg.get("mix_components_max", 1)),
                        float(run_cfg.get("component_weight_pareto_alpha", 0.0)),
                        float(run_cfg.get("token_noise_std", 0.0)),
                        float(run_cfg.get("token_noise_t_df", 0.0)),
                    )
                    if key not in probe_cache:
                        if probe_type == "clean_band":
                            probe = build_fixed_band_probe_split(
                                n=int(args.probe_size),
                                seq_len=key[3],
                                vocab_size=key[4],
                                c0=key[1],
                                o0=key[2],
                                min_step_frac=key[5],
                                allow_noncoprime=key[6],
                                noncoprime_prob=key[7],
                                seed=int(args.probe_seed + 97 * band_idx),
                            )
                        else:
                            probe = build_matched_band_probe_split(
                                n=int(args.probe_size),
                                seq_len=key[3],
                                vocab_size=key[4],
                                c0=key[1],
                                o0=key[2],
                                zipf_c=float(run_cfg.get("zipf_c", 1.3)),
                                zipf_o=float(run_cfg.get("zipf_o", 1.2)),
                                c_min=int(run_cfg.get("c_min", 5)),
                                min_step_frac=key[5],
                                allow_noncoprime=key[6],
                                noncoprime_prob=key[7],
                                mix_components_min=key[8],
                                mix_components_max=key[9],
                                component_weight_pareto_alpha=key[10],
                                token_noise_std=key[11],
                                token_noise_t_df=key[12],
                                seed=int(args.probe_seed + 197 * band_idx),
                            )
                        probe_cache[key] = probe
                    else:
                        probe = probe_cache[key]

                    out = run_feature_probe_once(
                        model=model,
                        probe_x=probe.x,
                        probe_y=probe.y,
                        state_labels=probe.state_labels,
                        c0=c0,
                        o0=o0,
                        device=device,
                        vocab_size=int(run_cfg.get("vocab_size", 1024)),
                        pos=int(args.position),
                        pca_k=int(args.pca_k),
                        control_seed=int(args.control_seed + 997 * band_idx),
                        batch_size=int(args.probe_batch_size),
                    )

                    base_meta = {
                        "task": row.get("task", run_cfg.get("task", "")),
                        "track": row.get("track", run_cfg.get("track", "")),
                        "variant_stage": row.get("variant_stage", ""),
                        "variant_index": int(row.get("variant_index", -1)),
                        "variant_combo": row.get("variant_combo", run_cfg.get("variant", "")),
                        "seed": int(row.get("seed", run_cfg.get("seed", -1))),
                        "B": int(row.get("B", run_cfg.get("B", -1))),
                        "checkpoint": int(step),
                        "checkpoint_selection_sources": ",".join(sorted(step_sources.get(int(step), set()))),
                        "checkpoint_align_key": primary_align_key,
                        "checkpoint_align_frac": float(primary_align_frac) if np.isfinite(primary_align_frac) else np.nan,
                        "checkpoint_progress_frac": float(step_progress) if np.isfinite(step_progress) else np.nan,
                        "checkpoint_progress_pct": int(step_progress_pct),
                        "final_checkpoint": int(final_ckpt_step),
                        "target_loss_used": float(target_loss) if target_loss is not None else np.nan,
                        "target_loss_metric_used": target_metric,
                        "target_loss_source": target_loss_source,
                        "target_loss_metric_source": target_metric_source,
                        "probe_band": f"{c0}:{o0}",
                        "probe_type": str(probe.metadata.get("probe_type", probe_type)),
                        "probe_matches_training_distribution": bool(probe.metadata.get("matches_training_distribution", False)),
                        "probe_anchor_component_forced": bool(probe.metadata.get("anchored_component_forced", True)),
                        "probe_mean_total_components": float(probe.metadata.get("mean_total_components", np.nan)),
                        "probe_mean_anchor_weight": float(probe.metadata.get("mean_anchor_weight", np.nan)),
                        "training_mix_components_min": int(run_cfg.get("mix_components_min", 1)),
                        "training_mix_components_max": int(run_cfg.get("mix_components_max", 1)),
                        "training_component_weight_pareto_alpha": float(run_cfg.get("component_weight_pareto_alpha", 0.0)),
                        "training_token_noise_std": float(run_cfg.get("token_noise_std", 0.0)),
                        "training_token_noise_t_df": float(run_cfg.get("token_noise_t_df", 0.0)),
                        "run_name": row.get("run_name", run_dir.name),
                        "run_dir": str(run_dir),
                    }

                    srow = {**base_meta, **out.summary_row}
                    crow = {**base_meta, **out.causal_row}
                    summary_rows.append(srow)
                    causal_rows.append(crow)
                    for prow in out.pca_rows:
                        pca_rows.append({**base_meta, **prow})

            checkpoint_manifest_rows.append(
                {
                    "task": row.get("task", run_cfg.get("task", "")),
                    "track": row.get("track", run_cfg.get("track", "")),
                    "variant_stage": row.get("variant_stage", ""),
                    "variant_index": int(row.get("variant_index", -1)),
                    "variant_combo": row.get("variant_combo", run_cfg.get("variant", "")),
                    "seed": int(row.get("seed", run_cfg.get("seed", -1))),
                    "B": int(row.get("B", run_cfg.get("B", -1))),
                    "run_name": row.get("run_name", run_dir.name),
                    "run_dir": str(run_dir),
                    "checkpoint": int(step),
                    "checkpoint_selection_sources": ",".join(sorted(step_sources.get(int(step), set()))),
                    "checkpoint_align_key": primary_align_key,
                    "checkpoint_align_frac": float(primary_align_frac) if np.isfinite(primary_align_frac) else np.nan,
                    "checkpoint_progress_frac": float(step_progress) if np.isfinite(step_progress) else np.nan,
                    "checkpoint_progress_pct": int(step_progress_pct),
                    "final_checkpoint": int(final_ckpt_step),
                    "target_loss_used": float(target_loss) if target_loss is not None else np.nan,
                    "target_loss_metric_used": target_metric,
                    "target_loss_source": target_loss_source,
                    "target_loss_metric_source": target_metric_source,
                }
            )

    summary_cols = [
        "task",
        "track",
        "variant_stage",
        "variant_index",
        "variant_combo",
        "seed",
        "B",
        "checkpoint",
        "checkpoint_selection_sources",
        "checkpoint_align_key",
        "checkpoint_align_frac",
        "checkpoint_progress_frac",
        "checkpoint_progress_pct",
        "final_checkpoint",
        "target_loss_used",
        "target_loss_metric_used",
        "target_loss_source",
        "target_loss_metric_source",
        "probe_band",
        "probe_type",
        "probe_matches_training_distribution",
        "probe_anchor_component_forced",
        "probe_mean_total_components",
        "probe_mean_anchor_weight",
        "training_mix_components_min",
        "training_mix_components_max",
        "training_component_weight_pareto_alpha",
        "training_token_noise_std",
        "training_token_noise_t_df",
        "H_peak",
        "H_gini",
        "E_peak",
        "E_gini",
        "Emb_peak",
        "Emb_gini",
        "H_mass_total",
        "E_mass_total",
        "Emb_mass_total",
        "H_dc_mass",
        "H_pos_mass",
        "dominant_freq",
        "pca_peak_mass_pc1",
        "base_probe_loss",
        "probe_support_min",
        "probe_support_mean",
        "run_name",
        "run_dir",
    ]
    causal_cols = [
        "task",
        "track",
        "variant_stage",
        "variant_index",
        "variant_combo",
        "seed",
        "B",
        "checkpoint",
        "checkpoint_selection_sources",
        "checkpoint_align_key",
        "checkpoint_align_frac",
        "checkpoint_progress_frac",
        "checkpoint_progress_pct",
        "final_checkpoint",
        "target_loss_used",
        "target_loss_metric_used",
        "target_loss_source",
        "target_loss_metric_source",
        "probe_band",
        "probe_type",
        "probe_matches_training_distribution",
        "probe_anchor_component_forced",
        "probe_mean_total_components",
        "probe_mean_anchor_weight",
        "training_mix_components_min",
        "training_mix_components_max",
        "training_component_weight_pareto_alpha",
        "training_token_noise_std",
        "training_token_noise_t_df",
        "keep_key_loss",
        "drop_key_loss",
        "keep_ctrl_loss",
        "drop_ctrl_loss",
        "delta_keep",
        "delta_drop",
        "delta_keep_ctrl",
        "delta_drop_ctrl",
        "delta_keep_vs_ctrl",
        "delta_drop_vs_ctrl",
        "keep_key_minus_base",
        "drop_key_minus_base",
        "keep_ctrl_minus_base",
        "drop_ctrl_minus_base",
        "key_freq",
        "ctrl_freq",
        "embedding_keep_key_loss",
        "embedding_drop_key_loss",
        "embedding_keep_ctrl_loss",
        "embedding_drop_ctrl_loss",
        "hidden_keep_key_loss",
        "hidden_drop_key_loss",
        "hidden_keep_ctrl_loss",
        "hidden_drop_ctrl_loss",
        "hidden_delta_keep",
        "hidden_delta_drop",
        "hidden_delta_keep_ctrl",
        "hidden_delta_drop_ctrl",
        "hidden_delta_keep_vs_ctrl",
        "hidden_delta_drop_vs_ctrl",
        "hidden_key_rank",
        "hidden_ctrl_rank",
        "run_name",
        "run_dir",
    ]
    pca_cols = [
        "task",
        "track",
        "variant_stage",
        "variant_index",
        "variant_combo",
        "seed",
        "B",
        "checkpoint",
        "checkpoint_selection_sources",
        "checkpoint_align_key",
        "checkpoint_align_frac",
        "checkpoint_progress_frac",
        "checkpoint_progress_pct",
        "final_checkpoint",
        "target_loss_used",
        "target_loss_metric_used",
        "target_loss_source",
        "target_loss_metric_source",
        "probe_band",
        "probe_type",
        "probe_matches_training_distribution",
        "probe_anchor_component_forced",
        "probe_mean_total_components",
        "probe_mean_anchor_weight",
        "training_mix_components_min",
        "training_mix_components_max",
        "training_component_weight_pareto_alpha",
        "training_token_noise_std",
        "training_token_noise_t_df",
        "pc_index",
        "pca_evr",
        "pca_peak_mass",
        "dominant_freq",
        "run_name",
        "run_dir",
    ]

    pd.DataFrame(summary_rows).reindex(columns=summary_cols).to_csv(out_dir / "feature_learning_summary.csv", index=False)
    pd.DataFrame(causal_rows).reindex(columns=causal_cols).to_csv(out_dir / "feature_causal_effects.csv", index=False)
    pd.DataFrame(pca_rows).reindex(columns=pca_cols).to_csv(out_dir / "feature_pca_summary.csv", index=False)
    pd.DataFrame(checkpoint_manifest_rows).to_csv(out_dir / "feature_probe_checkpoint_manifest.csv", index=False)
    pd.DataFrame(missing_runs).reindex(columns=["run_name", "reason"]).to_csv(out_dir / "feature_probe_missing_runs.csv", index=False)

    print(f"Wrote: {out_dir / 'feature_learning_summary.csv'}")
    print(f"Wrote: {out_dir / 'feature_causal_effects.csv'}")
    print(f"Wrote: {out_dir / 'feature_pca_summary.csv'}")
    print(f"Wrote: {out_dir / 'feature_probe_checkpoint_manifest.csv'}")
    print(f"Wrote: {out_dir / 'feature_probe_missing_runs.csv'}")


if __name__ == "__main__":
    main()
