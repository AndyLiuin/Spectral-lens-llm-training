from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _resolve_run_dir(row: pd.Series, source_ablation_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    run_name = row.get("run_name", "")
    if isinstance(run_name, str) and run_name.strip():
        candidates.append(source_ablation_dir / run_name.strip())
    run_dir = row.get("run_dir", "")
    if isinstance(run_dir, str) and run_dir.strip():
        raw = Path(run_dir.strip())
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((Path.cwd() / raw).resolve())
            candidates.append((source_ablation_dir.parent.parent / raw).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _filter_summary(
    df: pd.DataFrame,
    *,
    variant_stages: List[str],
    tracks: List[str],
    seeds: List[int],
    batches: List[int],
) -> pd.DataFrame:
    work = df.copy()
    if variant_stages:
        work = work[work["variant_stage"].astype(str).isin(variant_stages)]
    if tracks:
        work = work[work["track"].astype(str).isin(tracks)]
    if seeds:
        work = work[pd.to_numeric(work["seed"], errors="coerce").isin(seeds)]
    if batches:
        work = work[pd.to_numeric(work["B"], errors="coerce").isin(batches)]
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine selected-run variant ablations into one synthetic feature-learning ablation.")
    parser.add_argument("--source-ablation-dirs", type=str, required=True)
    parser.add_argument("--out-ablation-dir", type=str, required=True)
    parser.add_argument("--variant-stages", type=str, default="")
    parser.add_argument("--tracks", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--B-list", type=str, default="")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    source_ablation_dirs = [Path(p) for p in parse_list(args.source_ablation_dirs)]
    out_ablation_dir = Path(args.out_ablation_dir)
    if out_ablation_dir.exists() and any(out_ablation_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing to write into existing populated directory without --force: {out_ablation_dir}")

    variant_stages = parse_list(args.variant_stages)
    tracks = parse_list(args.tracks)
    seeds = parse_int_list(args.seeds) if str(args.seeds).strip() else []
    batches = parse_int_list(args.B_list) if str(args.B_list).strip() else []

    frames: List[pd.DataFrame] = []
    manifest_rows: List[dict] = []

    for order_idx, ablation_dir in enumerate(source_ablation_dirs):
        summary_csv = ablation_dir / "variant_concat_ablation_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary csv: {summary_csv}")
        df = pd.read_csv(summary_csv)
        df = _filter_summary(
            df,
            variant_stages=variant_stages,
            tracks=tracks,
            seeds=seeds,
            batches=batches,
        ).copy()
        if df.empty:
            continue
        resolved_dirs = []
        for _, row in df.iterrows():
            resolved = _resolve_run_dir(row, source_ablation_dir=ablation_dir)
            if resolved is None:
                raise FileNotFoundError(
                    f"Could not resolve run_dir for run '{row.get('run_name', '')}' from source '{ablation_dir}'"
                )
            resolved_dirs.append(str(resolved))
        df["run_dir"] = resolved_dirs
        df["source_ablation_dir"] = str(ablation_dir)
        df["source_summary_csv"] = str(summary_csv)
        df["source_order"] = int(order_idx)
        frames.append(df)
        manifest_rows.append(
            {
                "source_order": int(order_idx),
                "source_ablation_dir": str(ablation_dir),
                "source_summary_csv": str(summary_csv),
                "rows_kept": int(len(df)),
            }
        )

    if not frames:
        raise SystemExit("No rows remained after filtering source ablations.")

    combined = pd.concat(frames, ignore_index=True)
    key_cols = ["track", "variant_index", "variant_stage", "variant_combo", "seed", "B"]
    dup_mask = combined.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dup_rows = combined.loc[dup_mask, key_cols + ["source_ablation_dir", "run_name"]].sort_values(key_cols)
        raise SystemExit(
            "Duplicate variant rows detected across sources.\n"
            + dup_rows.to_string(index=False)
        )

    combined = combined.sort_values(["track", "variant_index", "seed", "B"]).reset_index(drop=True)
    out_ablation_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(out_ablation_dir / "variant_concat_ablation_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_ablation_dir / "variant_concat_ablation_combined_manifest.csv", index=False)
    protocol = {
        "source_ablation_dirs": [str(p) for p in source_ablation_dirs],
        "variant_stages": variant_stages,
        "tracks": tracks,
        "seeds": seeds,
        "batches": batches,
        "rows_written": int(len(combined)),
    }
    (out_ablation_dir / "variant_concat_ablation_combined_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_summary.csv'}")
    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_combined_manifest.csv'}")
    print(f"Wrote: {out_ablation_dir / 'variant_concat_ablation_combined_protocol.json'}")


if __name__ == "__main__":
    main()
