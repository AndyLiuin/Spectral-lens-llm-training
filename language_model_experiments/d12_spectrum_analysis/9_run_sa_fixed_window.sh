#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy.liu@yale.edu
#SBATCH --job-name=gpt_fixwin_spec
#SBATCH --output=logs/spec_fixwin_%A_%a.out
#SBATCH --error=logs/spec_fixwin_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
          TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
          LAYERS="${LAYERS:-0,3,6,9,11}"
          MATRICES="${MATRICES:-attn.c_proj.weight,mlp.c_proj.weight,attn.c_q.weight,attn.c_v.weight}"
          CHECKPOINT_DIR="$(resolve_path "${CHECKPOINT_DIR:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_const_loss/data_fineweb_const_loss/9_fixed_window_gpt2}")"
          VALIDATION_DATA="${VALIDATION_DATA:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
          OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_const_loss/data_fineweb_const_loss/9_fixed_window_gpt2/spectrum_data}")"
          PY_SCRIPT="${SCRIPT_DIR}/9_sa_fixed_window_1gpu.py"

          mkdir -p "${OUTPUT_DIR}"

          srun python "${PY_SCRIPT}" \
--checkpoint_dir "${CHECKPOINT_DIR}" \
--validation_data_path "${VALIDATION_DATA}" \
--output_dir "${OUTPUT_DIR}" \
--task_index "${TASK_ID}" \
--layers "${LAYERS}" \
--matrices "${MATRICES}" \
--seq_length 65536 \
--num_samples_grad 512 \
--num_samples_cov 512 \
--cov_batch_size 2 \
--grad_batch_size 2 \
--tail_start 30 \
--tail_finish 150
