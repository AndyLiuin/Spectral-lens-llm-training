#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_5_muon_streaming_paper_lr
#SBATCH --output=logs/5_muon_streaming_paper_lr_%A.out
#SBATCH --error=logs/5_muon_streaming_paper_lr_%A.err

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
PY_SCRIPT="5_muon_streaming_paper_lr.py"

# HuggingFace streaming configuration
HF_DATASET="HuggingFaceFW/fineweb"
HF_CONFIG="sample-100BT"
HF_VAL_CONFIG="sample-10BT"
HF_SPLIT="train"
BUFFER_TOKENS=20000000
EPOCH_TOKENS=100000000000

# Output directory
LEARNING_RATE=6e-4
MUON_LR_SCALE=0.2
OUT_DIR="/nfs/roberts/scratch/pi_jks79/zl664/Scaling_final/const_loss_run_gpt2s/5_muon_streaming_paper_lr/bs524288_lr${LEARNING_RATE}_h${MUON_LR_SCALE}"

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
STOP_MODE="const_loss"
LOSS_THRESHOLD=3.2

# Optimizer
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
echo "==========================================="
echo "Starting 1-GPU Training: Muon (paper LR scaling)"
echo "==========================================="
echo "Script: ${PY_SCRIPT}"
echo "Output: ${OUT_DIR}"
echo ""
echo "Data: Streaming from ${HF_DATASET}/${HF_CONFIG}"
echo "Val:  Streaming from ${HF_DATASET}/${HF_VAL_CONFIG}"
echo ""
echo "Model: ${MODEL}"
echo "Batch: batch_size=${BATCH_SIZE}, seq_len=${SEQ_LEN}, total_batch=${TOTAL_BATCH_SIZE}"
echo "Training: ${NUM_ITERS} iterations, early-stop threshold=${LOSS_THRESHOLD}"
echo ""
echo "Optimizer: AdamW+Muon with shared LR=${LEARNING_RATE}, Muon h=${MUON_LR_SCALE}, momentum=${MUON_MOMENTUM}"
echo "==========================================="
echo ""

CMD=(
  srun python "${PY_SCRIPT}"
  --hf_dataset "${HF_DATASET}"
  --hf_config "${HF_CONFIG}"
  --hf_val_config "${HF_VAL_CONFIG}"
  --hf_split "${HF_SPLIT}"
  --buffer_tokens ${BUFFER_TOKENS}
  --epoch_tokens ${EPOCH_TOKENS}
  --output_dir "${OUT_DIR}"
  --model "${MODEL}"
  --batch_size ${BATCH_SIZE}
  --sequence_length ${SEQ_LEN}
  --total_batch_size ${TOTAL_BATCH_SIZE}
  --num_iterations ${NUM_ITERS}
  --grad_clip ${GRAD_CLIP}
  --learning_rate ${LEARNING_RATE}
  --muon_lr_scale ${MUON_LR_SCALE}
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

# Add stopping threshold based on mode
if [[ "${STOP_MODE}" == "const_loss" && -n "${LOSS_THRESHOLD:-}" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ "${STOP_MODE}" == "epoch" && -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi

# Execute
"${CMD[@]}"
