#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy.liu@yale.edu
#SBATCH --job-name=gpt_adj_spec
#SBATCH --output=logs/spec_adj_%A_%a.out
#SBATCH --error=logs/spec_adj_%A_%a.err

set -euo pipefail

resolve_script_dir() {
  local script_path=""
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    local job_info
    job_info="$(scontrol show job -o "${SLURM_JOB_ID}" 2>/dev/null || true)"
    script_path="${job_info##* Command=}"
    script_path="${script_path%% *}"
    if [[ -n "${script_path}" ]]; then
      if [[ "${script_path}" != /* ]]; then
        script_path="${SLURM_SUBMIT_DIR:-$PWD}/${script_path}"
      fi
      (cd "$(dirname "${script_path}")" && pwd)
      return 0
    fi
  fi
  (cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
}

SCRIPT_DIR="$(resolve_script_dir)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]:-$0}")"
SA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${SA_ROOT}" "$1" ;;
  esac
}

count_checkpoints() {
  local checkpoint_dir="$1"
  find "${checkpoint_dir}" -maxdepth 1 -type f \( \
    -name 'checkpoint_step*.pt' -o \
    -name 'ckpt_step_*.pt' -o \
    -name 'checkpoint_target*.pt' \
  \) | wc -l | tr -d '[:space:]'
}

LAYERS="${LAYERS:-0,3,6,9,11}"
MATRICES="${MATRICES:-attn.c_proj.weight,mlp.c_proj.weight,attn.c_q.weight,attn.c_v.weight}"
CHECKPOINT_DIR="$(resolve_path "${CHECKPOINT_DIR:-data_fineweb_const_loss/16_adj_fp8lmhead_gpt2}")"
VALIDATION_DATA="${VALIDATION_DATA:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR:-${CHECKPOINT_DIR}/spectrum_data}")"
PY_SCRIPT="${SCRIPT_DIR}/16_adj_sa_fp8lmhead_1gpu.py"

mkdir -p "${OUTPUT_DIR}"

CHECKPOINT_COUNT="$(count_checkpoints "${CHECKPOINT_DIR}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CHECKPOINT_COUNT}" == "0" ]]; then
    echo "No checkpoints found in ${CHECKPOINT_DIR}" >&2
    exit 1
  fi

  ARRAY_SPEC="0-$((CHECKPOINT_COUNT - 1))"
  echo "Submitting ${CHECKPOINT_COUNT} spectrum-analysis jobs for ${CHECKPOINT_DIR}"
  echo "Output dir: ${OUTPUT_DIR}"

  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  VALIDATION_DATA="${VALIDATION_DATA}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LAYERS="${LAYERS}" \
  MATRICES="${MATRICES}" \
  exec sbatch \
    --chdir "${SCRIPT_DIR}" \
    --array "${ARRAY_SPEC}" \
    --output "${LOG_DIR}/spec_adj_%A_%a.out" \
    --error "${LOG_DIR}/spec_adj_%A_%a.err" \
    "${SCRIPT_PATH}"
fi

if [[ "${CHECKPOINT_COUNT}" != "0" && "${SLURM_ARRAY_TASK_COUNT:-1}" == "1" && "${CHECKPOINT_COUNT}" != "1" ]]; then
  echo "Warning: only one array task is active for ${CHECKPOINT_COUNT} checkpoints."
  echo "Run bash ${SCRIPT_PATH} to auto-submit the full array."
fi

module load miniconda
conda activate transform

echo "Job Start: ${SLURM_JOB_ID:-N/A}, Task: ${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

srun python "${PY_SCRIPT}" \
--checkpoint_dir "${CHECKPOINT_DIR}" \
--validation_data_path "${VALIDATION_DATA}" \
--output_dir "${OUTPUT_DIR}" \
--task_index "${TASK_ID}" \
--layers "${LAYERS}" \
--matrices "${MATRICES}"
