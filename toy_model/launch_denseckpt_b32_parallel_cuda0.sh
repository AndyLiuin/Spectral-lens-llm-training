#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="runs/modarith_a_seed0_steps5000_target25_core4_scaledsteps_b32_denseckpt_parallel_20260419/concat_batch_regime_selected_runs"
ABLATION_NAME="variant_concat_ablation_core4_target25_scaledsteps_b32_denseckpt_parallel"
SOURCE_BASE="runs/modarith_a_seed0_steps5000_target25_core4_scaledsteps_20260418/concat_batch_regime_selected_runs/variant_concat_ablation_core4_target25_scaledsteps"

mkdir -p logs

launch_one() {
    local tag="$1"
    local source_run_dir="$2"
    local log_path="logs/${tag}.log"
    echo "[launch] ${tag} -> ${log_path}"
    "$PYTHON_BIN" replay_single_selected_run.py \
        --source-run-dir "$source_run_dir" \
        --out-root "$OUT_ROOT" \
        --ablation-name "$ABLATION_NAME" \
        --device cuda:0 \
        >"$log_path" 2>&1 &
}

launch_one "denseckpt_b32_baseline_cuda0" \
    "${SOURCE_BASE}/track-a__D-32000__B-32__seed-0__var-baseline__steps-40000__task-mod_arith_lm__V-1024__mpool-last__h-7acec23145ea"
launch_one "denseckpt_b32_rope_cuda0" \
    "${SOURCE_BASE}/track-a__D-32000__B-32__seed-0__var-baseline+rope__steps-40000__task-mod_arith_lm__V-1024__mpool-last__h-f5f069e27c02"
launch_one "denseckpt_b32_muon_cuda0" \
    "${SOURCE_BASE}/track-a__D-32000__B-32__seed-0__var-baseline+rope+muon__steps-40000__task-mod_arith_lm__V-1024__mpool-last__h-671c4f172be9"
launch_one "denseckpt_b32_untie_cuda0" \
    "${SOURCE_BASE}/track-a__D-32000__B-32__seed-0__var-baseline+rope+muon+untie_embed__steps-40000__task-mod_arith_lm__V-1024__mpool-last__h-20a0cf816df1"

wait
echo "[launch] all four runs finished"
