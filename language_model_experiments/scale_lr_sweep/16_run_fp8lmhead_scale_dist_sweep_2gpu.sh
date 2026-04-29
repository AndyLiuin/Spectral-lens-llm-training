#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=sweep16_dist2g
#SBATCH --output=logs/sweep16_dist2g_%A.out
#SBATCH --error=logs/sweep16_dist2g_%A.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

module load miniconda
conda activate transform

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${SA_ROOT}" "$1" ;;
  esac
}
SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/gpt2_scale}"
mkdir -p "${SCRATCH_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL_PROFILE="${MODEL_PROFILE:-d24}"
PARALLEL_MODE="${PARALLEL_MODE:-auto}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-auto}"
COMPILE_MODE="${COMPILE_MODE:-auto}"
LOSS_FP32="${LOSS_FP32:-auto}"
VTE_ENDPOINT_K="${VTE_ENDPOINT_K:-0}"
if [[ -z "${SEQ_LEN:-}" ]]; then
  case "${MODEL_PROFILE}" in
    d36) SEQ_LEN=32768 ;;
    d48) SEQ_LEN=24576 ;;
    *) SEQ_LEN=65536 ;;
  esac
fi

TRAIN_PATTERN="${TRAIN_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/lr_sweep_dist/16_fp8lmhead_${MODEL_PROFILE}_dist2g}"
RUNGS="${RUNGS:-40000000 70000000 100000000}"
ETA="${ETA:-3}"
EXTRAP_TARGET="${EXTRAP_TARGET:-1000000000}"
PY_SCRIPT="${SCRIPT_DIR}/16_fp8lmhead_scale_dist_sweep.py"

set +e
torchrun --standalone --nproc_per_node=2 "${PY_SCRIPT}" \
  --train_pattern "${TRAIN_PATTERN}" \
  --val_pattern "${VAL_PATTERN}" \
  --output_dir "${OUT_DIR}" \
  --model_profile "${MODEL_PROFILE}" \
  --sequence_length "${SEQ_LEN}" \
  --parallel_mode "${PARALLEL_MODE}" \
  --activation_checkpointing "${ACTIVATION_CHECKPOINTING}" \
  --compile_mode "${COMPILE_MODE}" \
  --loss_fp32 "${LOSS_FP32}" \
  --vte_endpoint_k "${VTE_ENDPOINT_K}" \
  --rungs ${RUNGS} \
  --eta "${ETA}" \
  --extrapolation_target "${EXTRAP_TARGET}"
status=$?
set -e

if [[ ${status} -ne 0 && "${PARALLEL_MODE}" != "ddp" ]]; then
  FALLBACK_SEQ="${SEQ_LEN}"
  if [[ "${MODEL_PROFILE}" == "d48" ]]; then
    FALLBACK_SEQ=16384
  elif [[ "${MODEL_PROFILE}" == "d36" ]]; then
    FALLBACK_SEQ=24576
  fi
  echo "[WARN] FSDP sweep failed for variant 16, retrying with DDP at sequence_length=${FALLBACK_SEQ}"
  torchrun --standalone --nproc_per_node=2 "${PY_SCRIPT}" \
  --train_pattern "${TRAIN_PATTERN}" \
  --val_pattern "${VAL_PATTERN}" \
  --output_dir "${OUT_DIR}_fallback_ddp" \
  --model_profile "${MODEL_PROFILE}" \
  --sequence_length "${FALLBACK_SEQ}" \
  --vte_endpoint_k "${VTE_ENDPOINT_K}" \
  --parallel_mode ddp \
  --activation_checkpointing "${ACTIVATION_CHECKPOINTING}" \
  --compile_mode off \
  --loss_fp32 "${LOSS_FP32}" \
  --rungs ${RUNGS} \
  --eta "${ETA}" \
  --extrapolation_target "${EXTRAP_TARGET}"
fi
