#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_3_baseline_stream
#SBATCH --output=logs/3_baseline_streaming_%A.out
#SBATCH --error=logs/3_baseline_streaming_%A.err

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
PY_SCRIPT="3_baseline_streaming.py"

# HuggingFace streaming configuration
HF_DATASET="HuggingFaceFW/fineweb"
HF_CONFIG="sample-100BT"        # ~100B tokens for training
HF_VAL_CONFIG="sample-10BT"     # Use a different config for validation
HF_SPLIT="train"
BUFFER_TOKENS=20000000           # 20M tokens buffered in memory
EPOCH_TOKENS=100000000000        # 100B tokens = 1 epoch (for logging)

# Model architecture (GPT-2 Small)
VOCAB_SIZE=50304
BLOCK_SIZE=1024
N_LAYER=12
N_HEAD=6
N_EMBD=768

# Batching
GLOBAL_BATCH_SIZE=512
DEVICE_BATCH_SIZE=64
SEQ_LEN=1024

# Training (iteration-based)
NUM_ITERATIONS=60000
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.1

# LR schedule
WARMUP_ITERS=0
LEARNING_RATE_DECAY_FRAC=0.0

# Validation and checkpointing (step-based)
VAL_EVERY_STEPS=100         # Validate every N steps
CHECKPOINT_EVERY_STEPS=800  # Checkpoint every N steps
VAL_TOKENS=10485760

SEED=42

OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/3_baseline_stream_gpt2/bs${GLOBAL_BATCH_SIZE}_lr${LEARNING_RATE}"

# ==================== STOPPING CRITERIA ====================
STOP_MODE="const_loss"      # Options: "const_loss" or "epoch"

# const_loss mode: Stop when val_loss <= threshold
LOSS_THRESHOLD=3.2

# epoch mode: Stop at this fraction of an epoch (uncomment if using epoch mode)
# STOP_EPOCH_FRAC=0.5

# WandB
WANDB_PROJECT="gpt2-dynamics"

# ----------------------------
# Launch (Single GPU)
# ----------------------------
echo "==========================================="
echo "Starting 1-GPU Training: Baseline GPT-2 (Streaming)"
echo "==========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Data: Streaming from ${HF_DATASET}/${HF_CONFIG}"
echo "Val:  Streaming from ${HF_DATASET}/${HF_VAL_CONFIG}"
echo "Buffer: ${BUFFER_TOKENS} tokens in memory"
echo ""
echo "Architecture: Baseline GPT-2 (no modifications)"
echo "Model: GPT-2 (${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} dim)"
echo "Batch: ${DEVICE_BATCH_SIZE} sequences/GPU, ${GLOBAL_BATCH_SIZE} global"
echo ""
echo "Training: ${NUM_ITERATIONS} iterations, lr=${LEARNING_RATE}"
echo "Schedule: cosine, warmup_iters=${WARMUP_ITERS}, decay_frac=${LEARNING_RATE_DECAY_FRAC}"
echo "==========================================="
echo ""

# Build command with all arguments
CMD=(
  srun python "${PY_SCRIPT}"
  --hf_dataset "${HF_DATASET}"
  --hf_config "${HF_CONFIG}"
  --hf_val_config "${HF_VAL_CONFIG}"
  --hf_split "${HF_SPLIT}"
  --buffer_tokens ${BUFFER_TOKENS}
  --epoch_tokens ${EPOCH_TOKENS}
  --out_dir "${OUT_DIR}"
  --seed ${SEED}
  --vocab_size ${VOCAB_SIZE}
  --block_size ${BLOCK_SIZE}
  --n_layer ${N_LAYER}
  --n_head ${N_HEAD}
  --n_embd ${N_EMBD}
  --global_batch_size ${GLOBAL_BATCH_SIZE}
  --device_batch_size ${DEVICE_BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --num_iterations ${NUM_ITERATIONS}
  --learning_rate ${LEARNING_RATE}
  --weight_decay ${WEIGHT_DECAY}
  --warmup_iters ${WARMUP_ITERS}
  --learning_rate_decay_frac ${LEARNING_RATE_DECAY_FRAC}
  --val_every_steps ${VAL_EVERY_STEPS}
  --checkpoint_every_steps ${CHECKPOINT_EVERY_STEPS}
  --val_tokens ${VAL_TOKENS}
  --stop_mode "${STOP_MODE}"
  --wandb_project "${WANDB_PROJECT}"
  --compile
  --bf16
  --tf32
)

# Add stopping threshold/target based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
