#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_untie_embed_1gpu
#SBATCH --output=logs/untie_embed_1gpu_%A.out
#SBATCH --error=logs/untie_embed_1gpu_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="6_untie_embed.py"

# Data paths
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# Output directory
OUT_DIR="data_fineweb_const_loss/6_untie_gpt2"

# Model architecture (GPT-2 Small)
VOCAB_SIZE=50304
N_LAYER=12
N_HEAD=6
N_EMBD=768

# Batching
BATCH_SIZE=64
SEQ_LEN=1024
TOTAL_BATCH_SIZE=524288  # 512 * 1024 tokens

# Optimizer learning rates (3-optimizer setup for untied embeddings)
EMBED_LR=0.3      # Adam LR for input embeddings (wte)
HEAD_LR=0.002     # Adam LR for output head (lm_head)
MUON_LR=0.02      # Muon LR for hidden layers
MUON_MOMENTUM=0.95

# LR schedule
WARMUP_FRAC=0.0
WARMDOWN_FRAC=0.1  # 10% linear warmdown
LR_SCHEDULE="linear"
MIN_LR_RATIO=0.0

# Training
NUM_ITERS=30000
GRAD_CLIP=1.0
SEED=42

# Stopping criteria
STOP_MODE="const_loss"      # Options: "const_loss" or "epoch"
LOSS_THRESHOLD=3.3          # For const_loss mode
# STOP_EPOCH_FRAC=0.5       # For epoch mode (uncomment if needed)

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
echo "Starting 1-GPU Training: Untied Embeddings"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Model: GPT-2 (${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} dim)"
echo "Batch: ${BATCH_SIZE} sequences/GPU, ${TOTAL_BATCH_SIZE} tokens total"
echo "LRs: embed=${EMBED_LR}, head=${HEAD_LR}, muon=${MUON_LR}"
echo "Training: ${NUM_ITERS} iterations, threshold=${LOSS_THRESHOLD}"
echo "Schedule: ${WARMDOWN_FRAC} warmdown (${LR_SCHEDULE})"
echo "=========================================="
echo ""

# Build command with all arguments
CMD=(
  srun python "${PY_SCRIPT}"
  --train_pattern "${TRAIN_PATTERN}"
  --val_pattern "${VAL_PATTERN}"
  --output_dir "${OUT_DIR}"
  --vocab_size ${VOCAB_SIZE}
  --n_layer ${N_LAYER}
  --n_head ${N_HEAD}
  --n_embd ${N_EMBD}
  --batch_size ${BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --total_batch_size ${TOTAL_BATCH_SIZE}
  --embed_lr ${EMBED_LR}
  --head_lr ${HEAD_LR}
  --muon_lr ${MUON_LR}
  --muon_momentum ${MUON_MOMENTUM}
  --warmup_frac ${WARMUP_FRAC}
  --warmdown_frac ${WARMDOWN_FRAC}
  --lr_schedule ${LR_SCHEDULE}
  --min_lr_ratio ${MIN_LR_RATIO}
  --num_iterations ${NUM_ITERS}
  --grad_clip ${GRAD_CLIP}
  --seed ${SEED}
  --val_every_steps ${VAL_EVERY_STEPS}
  --val_tokens ${VAL_TOKENS}
  --checkpoint_every_steps ${CHECKPOINT_EVERY_STEPS}
  --stop_mode "${STOP_MODE}"
  --dtype ${DTYPE}
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
