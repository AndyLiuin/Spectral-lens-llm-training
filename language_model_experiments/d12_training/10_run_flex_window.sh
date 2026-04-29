#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_10_flex_window
#SBATCH --output=logs/10_flex_window_%A.out
#SBATCH --error=logs/10_flex_window_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="10_flex_window.py"

# Data paths
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# Batching (Flex Window with long sequences)
BATCH_SIZE=32
DEVICE_BATCH_SIZE=1
SEQ_LEN=65536

# Optimizer learning rates
EMBED_LR=0.6
HEAD_LR=0.008
MUON_LR=0.04
SCALAR_LR=0.04

# Muon momentum warmup (step-based)
MUON_MOMENTUM_INIT=0.85
MUON_MOMENTUM_FINAL=0.95
MUON_MOMENTUM_WARMUP_STEPS=125

# LR schedule
WARMUP_FRAC=0.0
WARMDOWN_FRAC=0.1

# Flex Window Schedule (step-based)
WINDOW_MIN=64
WINDOW_MAX=1792
WINDOW_WARMUP_STEPS=1000

# Training
NUM_ITERS=10000
GRAD_CLIP=1.0
SEED=42

OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/10_flex_window_gpt2/bs${BATCH_SIZE}_elr${EMBED_LR}_hlr${HEAD_LR}_mlr${MUON_LR}_slr${SCALAR_LR}"

# Stopping criteria
STOP_MODE="const_loss"      # Options: "const_loss" or "epoch"
LOSS_THRESHOLD=3.2          # Used when STOP_MODE="const_loss"
STOP_EPOCH_FRAC=""          # Set to e.g. "0.5" when STOP_MODE="epoch" (empty = disabled)

# Validation and checkpointing (step-based)
VAL_EVERY_STEPS=25
VAL_TOKENS=10485760
CHECKPOINT_EVERY_STEPS=100

# Numerics
DTYPE="bfloat16"

# WandB
WANDB_PROJECT="gpt2-dynamics"
WANDB_RUN_NAME="10_flex_window_gpt2"
WANDB_MODE="online"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "=========================================="
echo "Starting 1-GPU Training: Flex Window Attention"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Architecture: GPT-2 + Flex Window Attention"
echo "Batch: ${BATCH_SIZE} sequences total, ${DEVICE_BATCH_SIZE} per GPU"
echo "Seq Len: ${SEQ_LEN} tokens"
echo ""
echo "Stopping Criteria:"
echo "  - Loss threshold: ${LOSS_THRESHOLD}"
echo "  - Epoch fraction: ${STOP_EPOCH_FRAC:-disabled}"
echo ""
echo "Warmup (step-based):"
echo "  - Window: ${WINDOW_WARMUP_STEPS} steps"
echo "  - Muon momentum: ${MUON_MOMENTUM_WARMUP_STEPS} steps"
echo "=========================================="
echo ""

# Build command with all arguments
CMD=(
  srun python "${PY_SCRIPT}"
  --train_pattern "${TRAIN_PATTERN}"
  --val_pattern "${VAL_PATTERN}"
  --output_dir "${OUT_DIR}"
  --batch_size ${BATCH_SIZE}
  --device_batch_size ${DEVICE_BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --embed_lr ${EMBED_LR}
  --head_lr ${HEAD_LR}
  --muon_lr ${MUON_LR}
  --scalar_lr ${SCALAR_LR}
  --muon_momentum_init ${MUON_MOMENTUM_INIT}
  --muon_momentum_final ${MUON_MOMENTUM_FINAL}
  --muon_momentum_warmup_steps ${MUON_MOMENTUM_WARMUP_STEPS}
  --warmup_frac ${WARMUP_FRAC}
  --warmdown_frac ${WARMDOWN_FRAC}
  --window_min ${WINDOW_MIN}
  --window_max ${WINDOW_MAX}
  --window_warmup_steps ${WINDOW_WARMUP_STEPS}
  --num_iterations ${NUM_ITERS}
  --grad_clip ${GRAD_CLIP}
  --seed ${SEED}
  --val_every_steps ${VAL_EVERY_STEPS}
  --val_tokens ${VAL_TOKENS}
  --checkpoint_every_steps ${CHECKPOINT_EVERY_STEPS}
  --stop_mode "${STOP_MODE}"
  --dtype ${DTYPE}
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
  --compile
  --tensorcores
)

# Add stopping threshold/target based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
