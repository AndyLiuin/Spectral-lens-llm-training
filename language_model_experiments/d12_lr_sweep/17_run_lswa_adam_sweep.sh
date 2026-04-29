#!/bin/bash
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_jks79_zl664
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-5
# Update the array range if you change the number of entries in BATCH_SIZES.
#SBATCH --job-name=sweep_17_adam
#SBATCH --output=logs/sweep_17_adam_%A_%a.out
#SBATCH --error=logs/sweep_17_adam_%A_%a.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform_b200

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="17_lswa_adam_sweep.py"
OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/17_lswa_adam"

# Data paths (Modify if needed)
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# ASHA Sweep Settings
BATCH_SIZES=(1 2 4 8 16 32)
RUNGS=(30000000 70000000 100000000)
ETA=2
EXTRAPOLATION_TARGET=1000000000

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#BATCH_SIZES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} for ${#BATCH_SIZES[@]} batch sizes"
  exit 1
fi

CURRENT_BATCH_SIZE="${BATCH_SIZES[$TASK_ID]}"
TASK_OUT_DIR="${OUT_DIR}/bs${CURRENT_BATCH_SIZE}"

echo "=========================================="
echo "Starting ASHA Sweep: 17_lswa_adam (Long-Short Window Attention)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Task: ${TASK_ID}"
echo "Script: ${PY_SCRIPT}"
echo "Output: ${TASK_OUT_DIR}"
echo "Batch Sizes: ${BATCH_SIZES[*]}"
echo "Selected Batch Size: ${CURRENT_BATCH_SIZE}"
echo "Rungs: ${RUNGS[*]}"
echo "Eta: ${ETA}"
echo "Target: ${EXTRAPOLATION_TARGET}"
echo "LR sweep: widened upper multiplier for untuned Adam"
echo "=========================================="

python "${PY_SCRIPT}" \
    --train_pattern "${TRAIN_PATTERN}" \
    --val_pattern "${VAL_PATTERN}" \
    --output_dir "${TASK_OUT_DIR}" \
    --batch_size "${CURRENT_BATCH_SIZE}" \
    --rungs "${RUNGS[@]}" \
    --eta "${ETA}" \
    --extrapolation_target "${EXTRAPOLATION_TARGET}"
