#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_10_scale
#SBATCH --output=logs/10_scale_%A.out
#SBATCH --error=logs/10_scale_%A.err

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
      MODEL_PROFILE="${MODEL_PROFILE:-d12}"
      SEQ_LEN="${SEQ_LEN:-65536}"
      TRAIN_PATTERN="${TRAIN_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin}"
      VAL_PATTERN="${VAL_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
      OUT_DIR_DEFAULT="data_fineweb_const_loss/10_flex_window_${MODEL_PROFILE}_scale"
      OUT_DIR="$(resolve_path "${OUT_DIR:-${OUT_DIR_DEFAULT}}")"
      WANDB_PROJECT="${WANDB_PROJECT:-gpt2-dynamics}"
      WANDB_MODE="${WANDB_MODE:-online}"
      WANDB_RUN_NAME="${WANDB_RUN_NAME:-10_flex_window_scale}"
      PY_SCRIPT="${SCRIPT_DIR}/10_flex_window_scale.py"

      srun python "${PY_SCRIPT}" \
--train_pattern "${TRAIN_PATTERN}" \
--val_pattern "${VAL_PATTERN}" \
--output_dir "${OUT_DIR}" \
--model_profile "${MODEL_PROFILE}" \
--sequence_length "${SEQ_LEN}" \
--wandb_project "${WANDB_PROJECT}" \
--wandb_run_name "${WANDB_RUN_NAME}" \
--wandb_mode "${WANDB_MODE}" \
--compile \
--tensorcores
