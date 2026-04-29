#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_adam_streaming
#SBATCH --output=logs/gpt2_adam_streaming_%A.out
#SBATCH --error=logs/gpt2_adam_streaming_%A.err

set -euo pipefail
mkdir -p logs

module load miniconda
conda activate transform

# Redirect all caches to scratch (avoid home-dir quota issues)
SCRATCH_BASE="/nfs/roberts/scratch/pi_jks79/zl664"
export HF_HOME="${SCRATCH_BASE}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf_datasets_cache"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}/hf_cache"
export WANDB_DIR="${SCRATCH_BASE}/.cache/wandb"
export WANDB_CACHE_DIR="${SCRATCH_BASE}/.cache/wandb"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/.cache/pip"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
         "${WANDB_DIR}" "${TMPDIR}" "${PIP_CACHE_DIR}"

# Ensure streaming dependencies are installed
pip install --quiet tiktoken datasets

# ----------------------------
# Configuration
# ----------------------------
PY_SCRIPT="4_run_rope_streaming.py"

# HuggingFace streaming configuration
HF_DATASET="HuggingFaceFW/fineweb"
HF_CONFIG="sample-100BT"
HF_VAL_CONFIG="sample-10BT"
HF_SPLIT="train"
BUFFER_TOKENS=20000000
EPOCH_TOKENS=100000000000

# Model selector for new script
MODEL="d12"

# Batching (new script semantics)
# batch_size = per-device batch size in sequences
# total_batch_size = global batch size in TOKENS
BATCH_SIZE=64
SEQ_LEN=1024
TOTAL_BATCH_SIZE=524288

# Optimization
LEARNING_RATE=6e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0

# Schedule (new script semantics)
NUM_ITERS=50000
WARMUP_FRAC=0.0
WARMDOWN_FRAC=0.1
LR_SCHEDULE="linear"
MIN_LR_RATIO=0.0

# Stopping criteria
STOP_MODE="const_loss"
LOSS_THRESHOLD=3.2

# Validation and checkpointing
VAL_EVERY_STEPS=100
VAL_TOKENS=10485760
CHECKPOINT_EVERY_STEPS=800

# Output directory
OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/4_adam_only_streaming/bs512_lr${LEARNING_RATE}"

# Numerics
DTYPE="bfloat16"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "==========================================="
echo "Starting 1-GPU Training: AdamW only (Streaming)"
echo "==========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Data: Streaming from ${HF_DATASET}/${HF_CONFIG}"
echo "Val:  Streaming from ${HF_DATASET}/${HF_VAL_CONFIG}"
echo ""
echo "Model preset: ${MODEL}"
echo "Batch: device batch ${BATCH_SIZE}, seq len ${SEQ_LEN}, total batch ${TOTAL_BATCH_SIZE} tokens"
echo ""
echo "Optimizer: AdamW (LR=${LEARNING_RATE}, WD=${WEIGHT_DECAY})"
echo "Schedule: warmup_frac=${WARMUP_FRAC}, warmdown_frac=${WARMDOWN_FRAC}, type=${LR_SCHEDULE}, min_lr_ratio=${MIN_LR_RATIO}"
echo "Training: ${NUM_ITERS} iterations"
echo "==========================================="
echo ""

CMD=(
  srun python "${PY_SCRIPT}"
  --hf_dataset "${HF_DATASET}"
  --hf_config "${HF_CONFIG}"
  --hf_val_config "${HF_VAL_CONFIG}"
  --hf_split "${HF_SPLIT}"
  --buffer_tokens "${BUFFER_TOKENS}"
  --epoch_tokens "${EPOCH_TOKENS}"
  --output_dir "${OUT_DIR}"
  --model "${MODEL}"
  --batch_size "${BATCH_SIZE}"
  --sequence_length "${SEQ_LEN}"
  --total_batch_size "${TOTAL_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --num_iterations "${NUM_ITERS}"
  --warmup_frac "${WARMUP_FRAC}"
  --warmdown_frac "${WARMDOWN_FRAC}"
  --lr_schedule "${LR_SCHEDULE}"
  --min_lr_ratio "${MIN_LR_RATIO}"
  --val_every_steps "${VAL_EVERY_STEPS}"
  --val_tokens "${VAL_TOKENS}"
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}"
  --stop_mode "${STOP_MODE}"
  --dtype "${DTYPE}"
  --compile 1
  --tensorcores 1
)

if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

"${CMD[@]}"