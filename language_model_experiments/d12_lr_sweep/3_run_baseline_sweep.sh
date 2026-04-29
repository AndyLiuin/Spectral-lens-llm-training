#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=sweep_03
#SBATCH --output=logs/sweep_03_%A.out
#SBATCH --error=logs/sweep_03_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="3_baseline_sweep.py"
OUT_DIR="lr_sweep/3_baseline"

# Data paths (Modify if needed)
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# ASHA Sweep Settings
# Note: With seq_len=1024 (vs 65536 before), batch sizes scaled by 64x to maintain token budget
# Token budgets: 256K, 512K, 1M, 2M tokens (same as before)
BATCH_SIZES=(256 512 1024 2048)
RUNGS=(50000000 150000000 500000000)
ETA=3
EXTRAPOLATION_TARGET=1000000000

echo "=========================================="
echo "Starting ASHA Sweep: 3_baseline (No RoPE, Simple AdamW)"
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
