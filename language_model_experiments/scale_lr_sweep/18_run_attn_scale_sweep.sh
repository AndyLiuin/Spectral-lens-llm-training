#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=sweep_18
#SBATCH --output=logs/sweep_18_%A.out
#SBATCH --error=logs/sweep_18_%A.err

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
      PY_SCRIPT="${SCRIPT_DIR}/18_attn_scale_sweep.py"
      TRAIN_PATTERN="${TRAIN_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin}"
      VAL_PATTERN="${VAL_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
      OUT_DIR_DEFAULT="lr_sweep/18_attn_scale"
      OUT_DIR="$(resolve_path "${OUT_DIR:-${OUT_DIR_DEFAULT}}")"
      BATCH_SIZES=("${BATCH_SIZE_1:-4}" "${BATCH_SIZE_2:-8}" "${BATCH_SIZE_3:-16}" "${BATCH_SIZE_4:-32}")
      RUNGS=("${RUNG_1:-50000000}" "${RUNG_2:-150000000}" "${RUNG_3:-500000000}")
      ETA="${ETA:-3}"
      EXTRAPOLATION_TARGET="${EXTRAPOLATION_TARGET:-1000000000}"

      python "${PY_SCRIPT}" \
--train_pattern "${TRAIN_PATTERN}" \
--val_pattern "${VAL_PATTERN}" \
--output_dir "${OUT_DIR}" \
--batch_sizes "${BATCH_SIZES[@]}" \
--rungs "${RUNGS[@]}" \
--eta "${ETA}" \
--extrapolation_target "${EXTRAPOLATION_TARGET}"
