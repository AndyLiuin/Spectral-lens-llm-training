#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=sweep_8
#SBATCH --output=logs/sweep_8_%A.out
#SBATCH --error=logs/sweep_8_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="8_unet_sweep.py"
OUT_DIR="lr_sweep/8_unet"

# Data paths (Modify if needed)
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# ASHA Sweep Settings
# Note: With seq_len=1024 (vs 65536 before), batch sizes scaled by 64x to maintain token budget
# Token budgets: 32K, 64K, 128K, 256K tokens (0.5x, 1x, 2x, 4x baseline)
BATCH_SIZES=(32 64 128 256)
RUNGS=(50000000 150000000 500000000)
ETA=3
EXTRAPOLATION_TARGET=1000000000

echo "=========================================="
echo "Starting ASHA Sweep: 8_unet"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo "Batch Sizes: ${BATCH_SIZES[*]}"
echo "Rungs: ${RUNGS[*]}"
echo "Eta: ${ETA}"
echo "Target: ${EXTRAPOLATION_TARGET}"
echo "=========================================="

python "${PY_SCRIPT}" \
    --train_pattern "${TRAIN_PATTERN}" \
    --val_pattern "${VAL_PATTERN}" \
    --output_dir "${OUT_DIR}" \
    --batch_sizes "${BATCH_SIZES[@]}" \
    --rungs "${RUNGS[@]}" \
    --eta "${ETA}" \
    --extrapolation_target "${EXTRAPOLATION_TARGET}"
