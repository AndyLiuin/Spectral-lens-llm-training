#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_5_muon
#SBATCH --output=logs/5_muon_%A.out
#SBATCH --error=logs/5_muon_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="5_muon.py"

# Data paths
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# Output directory
OUT_DIR="data_fineweb_const_loss/5_muon_gpt2"

# Model
MODEL="d12"

# Batching
BATCH_SIZE=64
SEQ_LEN=1024
TOTAL_BATCH_SIZE=524288

# Training
NUM_ITERS=50000
GRAD_CLIP=1.0

# Stopping criteria
STOP_MODE="const_loss"      # Options: "const_loss" or "epoch"
LOSS_THRESHOLD=3.3          # For const_loss mode
# STOP_EPOCH_FRAC=0.5       # For epoch mode (uncomment if needed)

# Optimizer
EMBED_LR=0.0036
MUON_LR=0.00036
MUON_MOMENTUM=0.95
WEIGHT_DECAY=0.0

# LR schedule
WARMUP_FRAC=0.0
WARMDOWN_FRAC=0.1
LR_SCHEDULE="linear"
MIN_LR_RATIO=0.0

# Validation and checkpointing (step-based)
VAL_EVERY_STEPS=100
VAL_TOKENS=10485760
CHECKPOINT_EVERY_STEPS=400

# Numerics
DTYPE="bfloat16"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "=========================================="
echo "Starting 1-GPU Training: Muon Optimizer"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Model: ${MODEL}"
echo "Batch: batch_size=${BATCH_SIZE}, seq_len=${SEQ_LEN}, total_batch=${TOTAL_BATCH_SIZE}"
echo "Training: ${NUM_ITERS} iterations, early-stop threshold=${LOSS_THRESHOLD}"
echo "=========================================="
echo ""

CMD=(
  srun python "${PY_SCRIPT}"
  --input_bin "${TRAIN_PATTERN}"
  --input_val_bin "${VAL_PATTERN}"
  --output_dir "${OUT_DIR}"
  --model "${MODEL}"

  --batch_size ${BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --total_batch_size ${TOTAL_BATCH_SIZE}

  --num_iterations ${NUM_ITERS}
  --grad_clip ${GRAD_CLIP}
  --loss_threshold ${LOSS_THRESHOLD}

  --embed_lr ${EMBED_LR}
  --muon_lr ${MUON_LR}
  --muon_momentum ${MUON_MOMENTUM}
  --weight_decay ${WEIGHT_DECAY}

  --warmup_frac ${WARMUP_FRAC}
  --warmdown_frac ${WARMDOWN_FRAC}
  --lr_schedule "${LR_SCHEDULE}"
  --min_lr_ratio ${MIN_LR_RATIO}

  --val_every_steps ${VAL_EVERY_STEPS}
  --val_tokens ${VAL_TOKENS}
  --checkpoint_every_steps ${CHECKPOINT_EVERY_STEPS}
  --stop_mode "${STOP_MODE}"

  --dtype "${DTYPE}"
  --tensorcores 1
  --compile 1
  --flash 1
)

# Add stopping threshold/target based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
