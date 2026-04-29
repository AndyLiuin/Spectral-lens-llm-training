#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_4_rope_streaming
#SBATCH --output=logs/4_rope_streaming_%A.out
#SBATCH --error=logs/4_rope_streaming_%A.err

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
PY_SCRIPT="4_rope_fineweb_streaming.py"

# HuggingFace streaming configuration
HF_DATASET="HuggingFaceFW/fineweb"
HF_CONFIG="sample-100BT"
HF_VAL_CONFIG="sample-10BT"
HF_SPLIT="train"
BUFFER_TOKENS=20000000
EPOCH_TOKENS=100000000000

# Model architecture (GPT-2 Small with RoPE)
VOCAB_SIZE=50304
N_LAYER=12
N_HEAD=6
N_EMBD=768

# Batching
GLOBAL_BATCH_SIZE=512
DEVICE_BATCH_SIZE=64
SEQ_LEN=1024

# Optimization (Standard AdamW)
LEARNING_RATE=6e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0

# Schedule
NUM_ITERS=50000
WARMUP_ITERS=0
LR_DECAY_FRAC=0.0

# Stopping criteria
STOP_MODE="const_loss"
LOSS_THRESHOLD=3.2

# Validation and checkpointing
VAL_EVERY_STEPS=100
VAL_TOKENS=10485760
CHECKPOINT_EVERY_STEPS=800

# Output directory
OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/4_rope_streaming_gpt2/bs${GLOBAL_BATCH_SIZE}_lr${LEARNING_RATE}"

# Numerics
DTYPE="bfloat16"
SEED=42

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "==========================================="
echo "Starting 1-GPU Training: RoPE + AdamW (Streaming)"
echo "==========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Data: Streaming from ${HF_DATASET}/${HF_CONFIG}"
echo "Val:  Streaming from ${HF_DATASET}/${HF_VAL_CONFIG}"
echo ""
echo "Architecture: GPT-2 Small with RoPE"
echo "Model: ${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} dim"
echo "Batch: ${GLOBAL_BATCH_SIZE} global seqs w/ ${DEVICE_BATCH_SIZE} per GPU"
echo ""
echo "Optimizer: AdamW (LR=${LEARNING_RATE}, WD=${WEIGHT_DECAY})"
echo "Schedule: ${WARMUP_ITERS} warmup iters, decay to ${LR_DECAY_FRAC} of LR"
echo "Training: ${NUM_ITERS} iterations"
echo "==========================================="
echo ""

# Build command
CMD=(
  srun python "${PY_SCRIPT}"
  --hf_dataset "${HF_DATASET}"
  --hf_config "${HF_CONFIG}"
  --hf_val_config "${HF_VAL_CONFIG}"
  --hf_split "${HF_SPLIT}"
  --buffer_tokens ${BUFFER_TOKENS}
  --epoch_tokens ${EPOCH_TOKENS}
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
  --compile 1
  --tensorcores 1
)

# Add stopping threshold based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
