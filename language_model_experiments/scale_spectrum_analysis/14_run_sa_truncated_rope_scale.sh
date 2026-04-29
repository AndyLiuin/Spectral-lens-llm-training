#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-39
#SBATCH --job-name=sa_14_scale
#SBATCH --output=logs/sa_14_scale_%A_%a.out
#SBATCH --error=logs/sa_14_scale_%A_%a.err

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
          echo "Job Start: ${SLURM_JOB_ID:-N/A}, Task: ${SLURM_ARRAY_TASK_ID:-0}"
          MODEL_PROFILE="${MODEL_PROFILE:-d12}"
          SEQ_LEN="${SEQ_LEN:-65536}"
          TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
          LAYERS="${LAYERS:-auto}"
          MATRICES="${MATRICES:-attn.c_proj.weight,mlp.c_proj.weight,attn.c_q.weight,attn.c_v.weight}"
          CHECKPOINT_DIR="$(resolve_path "${CHECKPOINT_DIR:-data_fineweb_const_loss/14_truncated_rope_${MODEL_PROFILE}_scale}")"
          VALIDATION_DATA="${VALIDATION_DATA:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
          OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR:-${CHECKPOINT_DIR}/spectrum_data_${MODEL_PROFILE}_scale}")"
          PY_SCRIPT="${SCRIPT_DIR}/14_sa_truncated_rope_scale_1gpu.py"

          mkdir -p "${OUTPUT_DIR}"

          srun python "${PY_SCRIPT}" \
--checkpoint_dir "${CHECKPOINT_DIR}" \
--validation_data_path "${VALIDATION_DATA}" \
--output_dir "${OUTPUT_DIR}" \
--task_index "${TASK_ID}" \
--layers "${LAYERS}" \
--model_profile "${MODEL_PROFILE}" \
--seq_length "${SEQ_LEN}" \
--matrices "${MATRICES}"
