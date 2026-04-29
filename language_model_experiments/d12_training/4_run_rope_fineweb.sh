#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_4_rope_fineweb
#SBATCH --output=logs/4_rope_fineweb_%A.out
#SBATCH --error=logs/4_rope_fineweb_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="4_rope_fineweb.py"

# Data paths
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# Output directory
# OUT_DIR is defined after batch/LR vars so ${GLOBAL_BATCH_SIZE} and ${LEARNING_RATE} are available

# Model architecture (GPT-2 Small)
VOCAB_SIZE=50304
N_LAYER=12
N_HEAD=6
N_EMBD=768

# Batching
# GLOBAL_BATCH_SIZE is global sequence count = 512
# Total tokens per step = 512 * 1024 = 524,288
GLOBAL_BATCH_SIZE=512
DEVICE_BATCH_SIZE=64
SEQ_LEN=1024

# Optimization (Standard AdamW)
# Replaces the 4-LR/Muon setup
LEARNING_RATE=6e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0

# Schedule
NUM_ITERS=50000
WARMUP_ITERS=0
LR_DECAY_FRAC=0.0

# Stopping criteria
STOP_MODE="const_loss"      # Options: "const_loss" or "epoch"
LOSS_THRESHOLD=3.2
# STOP_EPOCH_FRAC=0.5       # For epoch mode (uncomment if needed)

# Validation and checkpointing
VAL_EVERY_STEPS=100
VAL_TOKENS=10485760
CHECKPOINT_EVERY_STEPS=800

# Output directory (defined here so GLOBAL_BATCH_SIZE and LEARNING_RATE are already set)
OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/4_rope_fineweb_gpt2/bs${GLOBAL_BATCH_SIZE}_lr${LEARNING_RATE}"

# Numerics
DTYPE="bfloat16"
SEED=42

# WandB
WANDB_PROJECT="gpt2-dynamics"
WANDB_RUN_NAME="4_rope_fineweb"
WANDB_MODE="online"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "=========================================="
echo "Starting 1-GPU Training: RoPE + AdamW + FineWeb"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Architecture: GPT-2 Small with RoPE"
echo "Model: ${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} dim"
echo "Batch: ${GLOBAL_BATCH_SIZE} global seqs w/ ${DEVICE_BATCH_SIZE} per GPU"
echo ""
echo "Optimizer: AdamW (LR=${LEARNING_RATE}, WD=${WEIGHT_DECAY})"
echo "Schedule: ${WARMUP_ITERS} warmup iters, decay to ${LR_DECAY_FRAC} of LR"
echo "Training: ${NUM_ITERS} iterations"
echo "=========================================="
echo ""

# Build command with all arguments
CMD=(
  srun
  python
  "${PY_SCRIPT}"
  --train_pattern "${TRAIN_PATTERN}"
  --val_pattern "${VAL_PATTERN}"
  --output_dir "${OUT_DIR}"
  --vocab_size "${VOCAB_SIZE}"
  --n_layer "${N_LAYER}"
  --n_head "${N_HEAD}"
  --n_embd "${N_EMBD}"
  --batch_size "${GLOBAL_BATCH_SIZE}"
  --device_batch_size "${DEVICE_BATCH_SIZE}"
  --sequence_length "${SEQ_LEN}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --num_iterations "${NUM_ITERS}"
  --warmup_iters "${WARMUP_ITERS}"
  --learning_rate_decay_frac "${LR_DECAY_FRAC}"
  --val_every_steps "${VAL_EVERY_STEPS}"
  --val_tokens "${VAL_TOKENS}"
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}"
  --stop_mode "${STOP_MODE}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
  --compile 1
  --tensorcores 1
)

# Add stopping threshold/target based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
