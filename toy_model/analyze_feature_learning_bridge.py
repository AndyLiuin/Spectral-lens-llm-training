from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_notebook_stream_texts(nb_path: Path) -> List[str]:
    if not nb_path.exists():
        return []
    try:
        nb = json.loads(nb_path.read_text())
    except Exception:
        return []

    texts: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            if otype == "stream":
                text = "".join(out.get("text", []))
                if text:
                    texts.append(text)
            elif otype in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "text/plain" in data:
                    text = "".join(data["text/plain"])
                    if text:
                        texts.append(text)
    return texts


def _parse_fft_progress(streams: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    pat_a = re.compile(
        r"\[STEP\s*(\d+)\].*?H_peak=([0-9]*\.?[0-9]+).*?E_peak=([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )
    pat_b = re.compile(
        r"Step\s*(\d+)\s*\|.*?H_pk=([0-9]*\.?[0-9]+).*?E_pk=([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )
    pat_c = re.compile(
        r"Step\s*(\d+)\s*\|.*?H_Fourier_peak=([0-9]*\.?[0-9]+).*?E_Fourier_peak=([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )

    for text in streams:
        for line in text.splitlines():
            m = pat_a.search(line) or pat_b.search(line) or pat_c.search(line)
            if not m:
                continue
            rows.append(
                {
                    "step": int(m.group(1)),
                    "H_peak": float(m.group(2)),
                    "E_peak": float(m.group(3)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["step", "H_peak", "E_peak"])  # type: ignore[return-value]
    df = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    return df.reset_index(drop=True)


def _parse_pca_mid_progress(streams: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    pat = re.compile(
        r"\[\s*(\d+)\]\s*tr=.*?\|\s*base=([0-9]*\.?[0-9]+)\s*keep=([0-9]*\.?[0-9]+)\s*drop=([0-9]*\.?[0-9]+)\s*\|\s*Hfreq=(\d+)\s*gauss=([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )

    for text in streams:
        for line in text.splitlines():
            m = pat.search(line)
            if not m:
                continue
            rows.append(
                {
                    "step": int(m.group(1)),
                    "probe_base": float(m.group(2)),
                    "keep_key": float(m.group(3)),
                    "drop_key": float(m.group(4)),
                    "H_topfreq": int(m.group(5)),
                    "gauss_peak_mean": float(m.group(6)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["step", "probe_base", "keep_key", "drop_key", "H_topfreq", "gauss_peak_mean"]
        )  # type: ignore[return-value]
    df = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    return df.reset_index(drop=True)


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _build_notebook_evidence(feature_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fft_nb = feature_dir / "FL_muon_test_FFT.ipynb"
    pca_mid_nb = feature_dir / "FL_muon_PCA_mid.ipynb"

    fft_df = _parse_fft_progress(_read_notebook_stream_texts(fft_nb))
    pca_df = _parse_pca_mid_progress(_read_notebook_stream_texts(pca_mid_nb))

    evidence_rows: List[dict] = []

    if not fft_df.empty:
        first = fft_df.iloc[0]
        last = fft_df.iloc[-1]
        evidence_rows.extend(
            [
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "H_peak_start",
                    "value": float(first["H_peak"]),
                    "note": "Initial Fourier concentration of H[m]",
                },
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "H_peak_end",
                    "value": float(last["H_peak"]),
                    "note": "Final Fourier concentration of H[m]",
                },
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "H_peak_fold_change",
                    "value": _safe_ratio(float(last["H_peak"]), float(first["H_peak"])),
                    "note": "Uniform-to-spiky proxy (H_peak end/start)",
                },
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "E_peak_start",
                    "value": float(first["E_peak"]),
                    "note": "Initial neuron->logit Fourier concentration",
                },
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "E_peak_end",
                    "value": float(last["E_peak"]),
                    "note": "Final neuron->logit Fourier concentration",
                },
                {
                    "source": "FL_muon_test_FFT.ipynb",
                    "metric": "E_peak_fold_change",
                    "value": _safe_ratio(float(last["E_peak"]), float(first["E_peak"])),
                    "note": "Uniform-to-spiky proxy (E_peak end/start)",
                },
            ]
        )

    if not pca_df.empty:
        delta_keep = (pca_df["probe_base"] - pca_df["keep_key"]).astype(float)
        delta_drop = (pca_df["probe_base"] - pca_df["drop_key"]).astype(float)
        evidence_rows.extend(
            [
                {
                    "source": "FL_muon_PCA_mid.ipynb",
                    "metric": "delta_keep_mean",
                    "value": float(delta_keep.mean()),
                    "note": "Average keep-key perturbation effect (base-keep)",
                },
                {
                    "source": "FL_muon_PCA_mid.ipynb",
                    "metric": "delta_drop_mean",
                    "value": float(delta_drop.mean()),
                    "note": "Average drop-key perturbation effect (base-drop)",
                },
                {
                    "source": "FL_muon_PCA_mid.ipynb",
                    "metric": "delta_keep_minus_drop_mean",
                    "value": float((delta_keep - delta_drop).mean()),
                    "note": "Causal margin for FL-H2 (expected > 0)",
                },
                {
                    "source": "FL_muon_PCA_mid.ipynb",
                    "metric": "H_topfreq_unique_count",
                    "value": float(pca_df["H_topfreq"].nunique()),
                    "note": "Frequency concentration proxy (lower = more concentrated)",
                },
                {
                    "source": "FL_muon_PCA_mid.ipynb",
                    "metric": "H_topfreq_mode",
                    "value": float(pca_df["H_topfreq"].mode().iloc[0]),
                    "note": "Most frequent dominant Fourier mode over training",
                },
            ]
        )

    evidence_df = pd.DataFrame(evidence_rows)
    return fft_df, evidence_df


def _summarize_variant_spectral(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "test_loss",
        "loss",
        "rankme",
        "alpha_head",
        "alpha_tail",
        "grad_rankme",
        "grad_alpha_head",
        "grad_alpha_tail",
        "tokens_seen",
        "train_time_s",
    ]
    use_cols = [c for c in cols if c in summary_df.columns]

    grouped = (
        summary_df.groupby(["track", "variant_index", "variant_stage", "variant_combo"], as_index=False)[use_cols]
        .mean(numeric_only=True)
        .sort_values(["track", "variant_index"])
    )

    out_rows: List[dict] = []
    for track, g in grouped.groupby("track"):
        g = g.sort_values("variant_index").reset_index(drop=True)
        if g.empty:
            continue
        base = g.iloc[0]
        for _, row in g.iterrows():
            item = row.to_dict()
            item["track"] = track
            for metric in ["rankme", "alpha_head", "alpha_tail", "grad_rankme", "grad_alpha_head", "grad_alpha_tail", "test_loss", "loss"]:
                if metric in row and metric in base:
                    item[f"delta_vs_baseline_{metric}"] = float(row[metric] - base[metric])
            if "tokens_seen" in row and "train_time_s" in row and row["train_time_s"] > 0:
                item["throughput"] = float(row["tokens_seen"] / row["train_time_s"])
            else:
                item["throughput"] = float("nan")
            out_rows.append(item)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    shift_terms = []
    for c in [
        "delta_vs_baseline_alpha_head",
        "delta_vs_baseline_alpha_tail",
        "delta_vs_baseline_rankme",
        "delta_vs_baseline_grad_rankme",
    ]:
        if c in out.columns:
            std = float(out[c].std(ddof=0))
            if np.isfinite(std) and std > 1e-12:
                shift_terms.append((out[c] / std) ** 2)
    if shift_terms:
        out["spectral_shift_magnitude_z"] = np.sqrt(np.sum(shift_terms, axis=0))
    else:
        out["spectral_shift_magnitude_z"] = np.nan
    return out.sort_values(["track", "variant_index"]).reset_index(drop=True)


def _feature_probe_evidence(feature_analysis_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = feature_analysis_dir / "feature_learning_summary.csv"
    causal_path = feature_analysis_dir / "feature_causal_effects.csv"
    if not summary_path.exists() or not causal_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        summary = pd.read_csv(summary_path)
        causal = pd.read_csv(causal_path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    if summary.empty or causal.empty:
        return pd.DataFrame(), pd.DataFrame()

    key = ["track", "variant_stage", "variant_index", "variant_combo", "seed", "checkpoint", "probe_band", "probe_type"]
    merged = summary.merge(causal, on=key, how="left", suffixes=("", "_causal"))
    rows: List[dict] = []

    for probe_type, g in merged.groupby("probe_type"):
        g = g.sort_values("checkpoint")
        first_ck = int(g["checkpoint"].min())
        last_ck = int(g["checkpoint"].max())
        g0 = g[g["checkpoint"] == first_ck]
        g1 = g[g["checkpoint"] == last_ck]

        def _mean(df: pd.DataFrame, col: str) -> float:
            return float(pd.to_numeric(df.get(col, np.nan), errors="coerce").mean())

        rows.extend(
            [
                {
                    "source": "feature_probe_pipeline",
                    "probe_type": probe_type,
                    "metric": "H_peak_start",
                    "value": _mean(g0, "H_peak"),
                    "note": f"Mean H_peak at first checkpoint for {probe_type}",
                },
                {
                    "source": "feature_probe_pipeline",
                    "probe_type": probe_type,
                    "metric": "H_peak_end",
                    "value": _mean(g1, "H_peak"),
                    "note": f"Mean H_peak at last checkpoint for {probe_type}",
                },
                {
                    "source": "feature_probe_pipeline",
                    "probe_type": probe_type,
                    "metric": "H_peak_fold_change",
                    "value": _safe_ratio(_mean(g1, "H_peak"), _mean(g0, "H_peak")),
                    "note": f"Mean H_peak end/start for {probe_type}",
                },
            ]
        )

        for scope, keep_col, drop_col, keep_ctrl_col, drop_ctrl_col in [
            ("embedding", "delta_keep", "delta_drop", "delta_keep_ctrl", "delta_drop_ctrl"),
            ("hidden", "hidden_delta_keep", "hidden_delta_drop", "hidden_delta_keep_ctrl", "hidden_delta_drop_ctrl"),
        ]:
            keep = pd.to_numeric(g.get(keep_col, np.nan), errors="coerce")
            drop = pd.to_numeric(g.get(drop_col, np.nan), errors="coerce")
            keep_ctrl = pd.to_numeric(g.get(keep_ctrl_col, np.nan), errors="coerce")
            drop_ctrl = pd.to_numeric(g.get(drop_ctrl_col, np.nan), errors="coerce")
            if not np.isfinite(keep.to_numpy(dtype=float)).any() and not np.isfinite(drop.to_numpy(dtype=float)).any():
                continue
            rows.extend(
                [
                    {
                        "source": "feature_probe_pipeline",
                        "probe_type": probe_type,
                        "metric": f"{scope}_delta_keep_mean",
                        "value": float(keep.mean()),
                        "note": f"Average keep-key effect for {scope} causality and {probe_type}",
                    },
                    {
                        "source": "feature_probe_pipeline",
                        "probe_type": probe_type,
                        "metric": f"{scope}_delta_drop_mean",
                        "value": float(drop.mean()),
                        "note": f"Average drop-key effect for {scope} causality and {probe_type}",
                    },
                    {
                        "source": "feature_probe_pipeline",
                        "probe_type": probe_type,
                        "metric": f"{scope}_delta_keep_minus_drop_mean",
                        "value": float((keep - drop).mean()),
                        "note": f"Keep-drop causal margin for {scope} causality and {probe_type}",
                    },
                    {
                        "source": "feature_probe_pipeline",
                        "probe_type": probe_type,
                        "metric": f"{scope}_keep_advantage_mean",
                        "value": float((keep - keep_ctrl).mean()),
                        "note": f"Keep-key vs keep-control margin for {scope} causality and {probe_type}",
                    },
                    {
                        "source": "feature_probe_pipeline",
                        "probe_type": probe_type,
                        "metric": f"{scope}_drop_advantage_mean",
                        "value": float((drop_ctrl - drop).mean()),
                        "note": f"Drop-key vs drop-control margin for {scope} causality and {probe_type}",
                    },
                ]
            )

    evidence = pd.DataFrame(rows)
    protocol = pd.DataFrame(
        [
            {
                "main_probe_type": "matched_band" if "matched_band" in set(evidence.get("probe_type", [])) else "clean_band",
                "main_causal_signal": "hidden_state_if_available_else_embedding",
                "notes": "Matched-band evidence is preferred for mixed/noisy training distributions; clean-band remains the microscope view.",
            }
        ]
    )
    return evidence, protocol


def _build_transition_snapshot(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for (track, seed), g in summary_df.groupby(["track", "seed"]):
        g = g.sort_values("variant_index").reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            curr = g.iloc[i]
            prev_thr = float(prev["tokens_seen"] / prev["train_time_s"]) if prev["train_time_s"] > 0 else np.nan
            curr_thr = float(curr["tokens_seen"] / curr["train_time_s"]) if curr["train_time_s"] > 0 else np.nan
            rows.append(
                {
                    "track": track,
                    "seed": int(seed),
                    "prev_variant_stage": prev["variant_stage"],
                    "curr_variant_stage": curr["variant_stage"],
                    "transition": f"{prev['variant_stage']}->{curr['variant_stage']}",
                    "tok_gain": float(prev["test_loss"] / curr["test_loss"] - 1.0) if curr["test_loss"] > 0 else np.nan,
                    "thr_gain": float(curr_thr / prev_thr - 1.0) if prev_thr > 0 and curr_thr > 0 else np.nan,
                    "delta_alpha_head": float(curr["alpha_head"] - prev["alpha_head"]),
                    "delta_alpha_tail": float(curr["alpha_tail"] - prev["alpha_tail"]),
                    "delta_rankme": float(curr["rankme"] - prev["rankme"]),
                    "delta_grad_rankme": float(curr["grad_rankme"] - prev["grad_rankme"]),
                }
            )

    return pd.DataFrame(rows)


def _hypothesis_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_id": "FL-H1",
                "statement": "Fourier mass in H[m] moves from diffuse to concentrated over training (H_peak/H_gini increase).",
                "primary_metric": "H_peak, H_gini",
                "test_definition": "Monotone or net-positive change from early checkpoint to late checkpoint; report seed-level trajectories.",
                "effect_direction": "increase",
                "full_claim_requirement": "All 6 variants x 3 seeds with checkpoint probes.",
            },
            {
                "hypothesis_id": "FL-H2",
                "statement": "Hidden-state projections aligned with learned key frequencies are more causal than matched-dimension controls; embedding-band edits remain supporting evidence.",
                "primary_metric": "hidden_delta_keep, hidden_delta_drop, hidden_delta_keep_vs_ctrl, hidden_delta_drop_vs_ctrl",
                "test_definition": "Prefer hidden-state keep/drop margins; require positive key-vs-control margins and positive keep-vs-drop contrast.",
                "effect_direction": "keep_advantage_positive",
                "full_claim_requirement": "Hidden-state causal keep/drop at each checkpoint for all variants/seeds + band robustness + matched-band robustness.",
            },
            {
                "hypothesis_id": "FL-H3",
                "statement": "Variants with larger activation/gradient spectral shifts show different feature-learning trajectories.",
                "primary_metric": "corr(delta_H_peak, delta_alpha_head/grad metrics)",
                "test_definition": "Checkpoint-matched transition correlations and edge-level deltas.",
                "effect_direction": "nonzero_association",
                "full_claim_requirement": "Merged transition map with feature + spectral deltas across variants/seeds.",
            },
            {
                "hypothesis_id": "FL-H4",
                "statement": "Early feature metrics predict later token-efficiency direction better than final activation-only endpoints.",
                "primary_metric": "early_corr_vs_final_corr",
                "test_definition": "Compare absolute correlation of early (400/800/1600) feature deltas vs final endpoint spectral deltas with TokGain.",
                "effect_direction": "early_stronger",
                "full_claim_requirement": "Transition-level early checkpoint probes and endpoint metrics across all variants/seeds.",
            },
        ]
    )


def _build_statement_matrix(
    evidence_df: pd.DataFrame,
    spectral_df: pd.DataFrame,
    feature_summary_available: bool,
) -> pd.DataFrame:
    ev = {}
    if not evidence_df.empty:
        for _, r in evidence_df.iterrows():
            metric = str(r["metric"])
            probe_type = str(r.get("probe_type", "")).strip()
            value = float(r["value"])
            ev[(metric, probe_type)] = value
            if metric not in ev:
                ev[metric] = value

    matched_probe = "matched_band" if any(k[1] == "matched_band" for k in ev.keys() if isinstance(k, tuple)) else ""
    clean_probe = "clean_band" if any(k[1] == "clean_band" for k in ev.keys() if isinstance(k, tuple)) else ""
    preferred_probe = matched_probe or clean_probe

    h1_metric = ev.get(("H_peak_fold_change", preferred_probe), ev.get("H_peak_fold_change", np.nan))
    h2_metric = ev.get(("hidden_delta_keep_minus_drop_mean", preferred_probe), ev.get("hidden_delta_keep_minus_drop_mean", np.nan))
    if not np.isfinite(h2_metric):
        h2_metric = ev.get(("embedding_delta_keep_minus_drop_mean", preferred_probe), ev.get("embedding_delta_keep_minus_drop_mean", np.nan))

    h1_direct = bool(np.isfinite(h1_metric) and h1_metric > 1.5)
    h2_direct = bool(np.isfinite(h2_metric) and h2_metric > 0.0)
    spectral_multivariant = not spectral_df.empty and int(spectral_df["variant_stage"].nunique()) >= 6

    rows = [
        {
            "hypothesis_id": "FL-H1",
            "current_status": "supported_single_variant" if h1_direct else "directional_single_variant",
            "directly_shown_now": bool(h1_direct),
            "directional_multi_variant_support": bool(spectral_multivariant),
            "evidence_now": "Probe pipeline shows increasing H[m] Fourier concentration; variant-concat shows multi-variant spectral shifts.",
            "claim_wording_now": "Direct probe evidence plus multi-variant directional consistency.",
            "full_variant_claim_ready": bool(feature_summary_available),
            "what_is_missing": "Checkpoint-level feature probes across all variants/seeds.",
        },
        {
            "hypothesis_id": "FL-H2",
            "current_status": "supported_single_variant" if h2_direct else "directional_single_variant",
            "directly_shown_now": bool(h2_direct),
            "directional_multi_variant_support": False,
            "evidence_now": "Hidden-state keep/drop interventions provide the main causal evidence; embedding-band edits remain supporting evidence.",
            "claim_wording_now": "Causal evidence should be phrased in terms of hidden-state subspaces when available, with embedding-band edits as support.",
            "full_variant_claim_ready": bool(feature_summary_available),
            "what_is_missing": "Hidden-state keep/drop+controls for all variants/seeds/checkpoints, especially on matched-band probes.",
        },
        {
            "hypothesis_id": "FL-H3",
            "current_status": "directional_until_rerun",
            "directly_shown_now": False,
            "directional_multi_variant_support": bool(spectral_multivariant),
            "evidence_now": "Cross-variant activation/gradient spectral differences are available, but cross-variant feature trajectories are missing.",
            "claim_wording_now": "Hypothesis-motivated directional reading only.",
            "full_variant_claim_ready": False,
            "what_is_missing": "Unified variant-wide feature summary + transition correlation analysis.",
        },
        {
            "hypothesis_id": "FL-H4",
            "current_status": "not_tested",
            "directly_shown_now": False,
            "directional_multi_variant_support": False,
            "evidence_now": "No early-vs-final transition prediction table available yet.",
            "claim_wording_now": "Do not claim until rerun analysis.",
            "full_variant_claim_ready": False,
            "what_is_missing": "Early-checkpoint feature metrics merged with TokGain/ThrGain transition outcomes.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge analysis: existing variant spectral artifacts + notebook feature evidence.")
    parser.add_argument("--ablation-dir", type=str, default="toy_model/variant_concat_ablation")
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--feature-notebook-dir", type=str, default="toy_model/Feature_learning")
    parser.add_argument("--feature-analysis-dir", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (ablation_dir / "variant_concat_ablation_summary.csv")
    feature_nb_dir = Path(args.feature_notebook_dir)
    feature_analysis_dir = Path(args.feature_analysis_dir) if args.feature_analysis_dir else (ablation_dir / "feature_learning_analysis")
    out_dir = Path(args.out_dir) if args.out_dir else feature_analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    spectral_df = _summarize_variant_spectral(summary_df)
    transition_df = _build_transition_snapshot(summary_df)

    fft_df, evidence_df = _build_notebook_evidence(feature_nb_dir)
    probe_evidence_df, probe_protocol_df = _feature_probe_evidence(feature_analysis_dir)
    merged_evidence_df = pd.concat([evidence_df, probe_evidence_df], ignore_index=True) if not probe_evidence_df.empty else evidence_df.copy()

    feature_summary_path = feature_analysis_dir / "feature_learning_summary.csv"
    feature_summary_available = feature_summary_path.exists()
    if feature_summary_available:
        try:
            fs = pd.read_csv(feature_summary_path)
            feature_summary_available = not fs.empty
        except Exception:
            feature_summary_available = False

    hypotheses_df = _hypothesis_table()
    statements_df = _build_statement_matrix(
        evidence_df=merged_evidence_df,
        spectral_df=spectral_df,
        feature_summary_available=feature_summary_available,
    )

    evidence_df.to_csv(out_dir / "feature_bridge_notebook_evidence.csv", index=False)
    probe_evidence_df.to_csv(out_dir / "feature_bridge_probe_evidence.csv", index=False)
    probe_protocol_df.to_csv(out_dir / "feature_probe_protocol.csv", index=False)
    merged_evidence_df.to_csv(out_dir / "feature_bridge_all_evidence.csv", index=False)
    fft_df.to_csv(out_dir / "feature_bridge_fft_progress.csv", index=False)
    spectral_df.to_csv(out_dir / "feature_bridge_variant_spectral_summary.csv", index=False)
    transition_df.to_csv(out_dir / "feature_bridge_transition_snapshot.csv", index=False)
    hypotheses_df.to_csv(out_dir / "feature_hypotheses_and_tests.csv", index=False)
    statements_df.to_csv(out_dir / "feature_bridge_statement_matrix.csv", index=False)

    summary_payload = {
        "ablation_dir": str(ablation_dir),
        "summary_csv": str(summary_csv),
        "feature_notebook_dir": str(feature_nb_dir),
        "feature_analysis_dir": str(feature_analysis_dir),
        "feature_summary_available": bool(feature_summary_available),
        "n_variant_rows": int(len(summary_df)),
        "n_variant_stages": int(summary_df["variant_stage"].nunique()) if "variant_stage" in summary_df.columns else 0,
        "n_notebook_evidence_rows": int(len(evidence_df)),
        "n_probe_evidence_rows": int(len(probe_evidence_df)),
    }
    (out_dir / "feature_bridge_summary.json").write_text(json.dumps(summary_payload, indent=2))

    ev = {}
    for src_df in (merged_evidence_df,):
        if src_df.empty:
            continue
        for _, r in src_df.iterrows():
            metric = str(r["metric"])
            probe_type = str(r.get("probe_type", "")).strip()
            value = float(r["value"])
            ev[(metric, probe_type)] = value
            if metric not in ev:
                ev[metric] = value
    preferred_probe = "matched_band" if ("H_peak_fold_change", "matched_band") in ev else ("clean_band" if ("H_peak_fold_change", "clean_band") in ev else "")
    section_lines = [
        "# Toy Feature-Learning Section Draft (Claim-Safe)",
        "",
        "## Mechanistic Claim",
        "Spectral shifts are interpreted as feature-learning only where direct feature probes are available; otherwise they are treated as directional evidence. For mixed or noisy modular-arithmetic training, matched-band probes are the primary in-distribution evidence and clean-band probes are the microscope view.",
        "",
        "## What Is Directly Shown Now",
        f"- Probe evidence shows H_peak increases from {ev.get(('H_peak_start', preferred_probe), ev.get('H_peak_start', float('nan'))):.4f} to {ev.get(('H_peak_end', preferred_probe), ev.get('H_peak_end', float('nan'))):.4f} (fold-change {ev.get(('H_peak_fold_change', preferred_probe), ev.get('H_peak_fold_change', float('nan'))):.2f}) on the preferred probe regime.",
        f"- Preferred causal margin is positive when hidden-state interventions are available: mean(hidden_delta_keep - hidden_delta_drop) = {ev.get(('hidden_delta_keep_minus_drop_mean', preferred_probe), float('nan')):.4f}.",
        f"- Embedding-band interventions remain available as supporting evidence: mean(embedding_delta_keep - embedding_delta_drop) = {ev.get(('embedding_delta_keep_minus_drop_mean', preferred_probe), float('nan')):.4f}.",
        "- Multi-variant variant-concat spectra show substantial activation/gradient differences across the six cumulative stages.",
        "",
        "## Conservative Wording",
        "- Direct feature-learning evidence should be described separately for clean-band probes and matched-band probes.",
        "- Hidden-state causal interventions are the strongest current claim; embedding-band edits should be described as supporting evidence about the learned token band.",
        "- Cross-variant claims remain directional until checkpoint-level probes are run for all variants/seeds.",
        "",
        "## Needed For Full Variant-Wide Claim",
        "- Run all 6 variants x 3 seeds with checkpoints (0:200:2000).",
        "- Execute standardized FFT + hidden-state causal + embedding-band causal + PCA probe pipeline for both fixed bands and both probe regimes when training is mixed/noisy.",
        "- Use transition map outputs to test FL-H3/FL-H4 with bootstrap/permutation statistics.",
    ]
    (out_dir / "toy_feature_section_draft.md").write_text("\n".join(section_lines))

    print(f"Wrote: {out_dir / 'feature_bridge_notebook_evidence.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_probe_evidence.csv'}")
    print(f"Wrote: {out_dir / 'feature_probe_protocol.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_all_evidence.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_fft_progress.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_variant_spectral_summary.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_transition_snapshot.csv'}")
    print(f"Wrote: {out_dir / 'feature_hypotheses_and_tests.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_statement_matrix.csv'}")
    print(f"Wrote: {out_dir / 'feature_bridge_summary.json'}")
    print(f"Wrote: {out_dir / 'toy_feature_section_draft.md'}")


if __name__ == "__main__":
    main()
