#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_12_better_window
#SBATCH --output=logs/12_better_window_%A.out
#SBATCH --error=logs/12_better_window_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="12_better_window.py"

# Data paths
TRAIN_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin"
VAL_PATTERN="/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_*.bin"

# Output directory
OUT_DIR="data_fineweb_const_loss/12_better_window_gpt2"

VOCAB_SIZE=50304
N_LAYER=12
N_HEAD=6
N_EMBD=768

BATCH_SIZE=8                    
DEVICE_BATCH_SIZE=1            
SEQ_LEN=65536                   

# Optimizer learning rates (4-optimizer setup for VTE)
EMBED_LR=0.6       # Adam LR for input embeddings (wte + vte)
HEAD_LR=0.008      # Adam LR for output head (lm_head)
MUON_LR=0.04       # Muon LR for matrix parameters
SCALAR_LR=0.04     # Adam LR for scalar parameters (lambdas, skip_weights)

# Muon momentum warmup
MUON_MOMENTUM_INIT=0.85
MUON_MOMENTUM_FINAL=0.95
MUON_MOMENTUM_WARMUP=500

# LR schedule
WARMUP_FRAC=0.0
WARMDOWN_FRAC=0.1  # 10% linear warmdown

# Training
NUM_ITERS=20000
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

# WandB
WANDB_PROJECT="gpt2-dynamics"
WANDB_RUN_NAME="12_better_window"
WANDB_MODE="online"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "=========================================="
echo "Starting 1-GPU Training: VTE Architecture"
echo "=========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Architecture: VTE (Value Token Embedding + Skip Connections + Flex Attention)"
echo "Model: GPT-2 (${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} dim)"
echo "Batch: ${BATCH_SIZE} sequences total, ${DEVICE_BATCH_SIZE} per GPU, ${SEQ_LEN} tokens/seq"
echo "Total tokens/step: $((BATCH_SIZE * SEQ_LEN))"
echo ""
echo "4-Optimizer Setup:"
echo "  - Embeddings (wte+vte) (Adam): ${EMBED_LR}"
echo "  - Output Head (Adam): ${HEAD_LR}"
echo "  - Matrix Params (Muon): ${MUON_LR}, momentum ${MUON_MOMENTUM_INIT}→${MUON_MOMENTUM_FINAL}"
echo "  - Scalar Params (Adam): ${SCALAR_LR}"
echo ""
echo "Training: ${NUM_ITERS} iterations, threshold=${LOSS_THRESHOLD}"
echo "Schedule: ${WARMDOWN_FRAC} warmdown"
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
  --device_batch_size ${DEVICE_BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --embed_lr ${EMBED_LR}
  --head_lr ${HEAD_LR}
  --muon_lr ${MUON_LR}
  --scalar_lr ${SCALAR_LR}
  --muon_momentum_init ${MUON_MOMENTUM_INIT}
  --muon_momentum_final ${MUON_MOMENTUM_FINAL}
  --muon_momentum_warmup_steps ${MUON_MOMENTUM_WARMUP}
  --warmup_frac ${WARMUP_FRAC}
  --warmdown_frac ${WARMDOWN_FRAC}
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
  --use_cudnn_attn
)

# Add stopping threshold/target based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
