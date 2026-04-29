#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=gpt2_18_attn_scale
#SBATCH --output=logs/18_attn_scale_%A.out
#SBATCH --error=logs/18_attn_scale_%A.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

module load miniconda
conda activate transform

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${SA_ROOT}" "$1" ;;
  esac
}
PY_SCRIPT="${SCRIPT_DIR}/18_attn_scale.py"
TRAIN_PATTERN="${TRAIN_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-/nfs/roberts/project/pi_jks79/zl664/Scaling_gpt2s/gpt_main/fineweb_10B/fineweb_val_000000.bin}"
OUT_DIR_DEFAULT="data_fineweb_const_loss/18_attn_scale_gpt2"
OUT_DIR="$(resolve_path "${OUT_DIR:-${OUT_DIR_DEFAULT}}")"
VOCAB_SIZE="${VOCAB_SIZE:-50304}"
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-6}"
N_EMBD="${N_EMBD:-768}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-65536}"
EMBED_LR="${EMBED_LR:-0.6}"
HEAD_LR="${HEAD_LR:-0.008}"
MUON_LR="${MUON_LR:-0.04}"
SCALAR_LR="${SCALAR_LR:-0.04}"
MUON_MOMENTUM_INIT="${MUON_MOMENTUM_INIT:-0.85}"
MUON_MOMENTUM_FINAL="${MUON_MOMENTUM_FINAL:-0.95}"
MUON_MOMENTUM_WARMUP="${MUON_MOMENTUM_WARMUP:-300}"
WARMUP_FRAC="${WARMUP_FRAC:-0.0}"
WARMDOWN_FRAC="${WARMDOWN_FRAC:-0.1}"
WINDOW_MIN="${WINDOW_MIN:-64}"
WINDOW_MAX="${WINDOW_MAX:-1792}"
WINDOW_WARMUP_STEPS="${WINDOW_WARMUP_STEPS:-3000}"
NUM_ITERS="${NUM_ITERS:-10000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-42}"
STOP_MODE="${STOP_MODE:-const_loss}"
LOSS_THRESHOLD="${LOSS_THRESHOLD:-3.3}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-100}"
VAL_TOKENS="${VAL_TOKENS:-10485760}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-400}"
DTYPE="${DTYPE:-bfloat16}"
WANDB_PROJECT="${WANDB_PROJECT:-gpt2-dynamics}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-18_attn_scale_gpt2}"
WANDB_MODE="${WANDB_MODE:-online}"

CMD=(
  srun python "${PY_SCRIPT}"
  --train_pattern "${TRAIN_PATTERN}"
  --val_pattern "${VAL_PATTERN}"
  --output_dir "${OUT_DIR}"
  --vocab_size "${VOCAB_SIZE}"
  --n_layer "${N_LAYER}"
  --n_head "${N_HEAD}"
  --n_embd "${N_EMBD}"
  --batch_size "${BATCH_SIZE}"
  --device_batch_size "${DEVICE_BATCH_SIZE}"
  --sequence_length "${SEQ_LEN}"
  --embed_lr "${EMBED_LR}"
  --head_lr "${HEAD_LR}"
  --muon_lr "${MUON_LR}"
  --scalar_lr "${SCALAR_LR}"
  --muon_momentum_init "${MUON_MOMENTUM_INIT}"
  --muon_momentum_final "${MUON_MOMENTUM_FINAL}"
  --muon_momentum_warmup_steps "${MUON_MOMENTUM_WARMUP}"
  --warmup_frac "${WARMUP_FRAC}"
  --warmdown_frac "${WARMDOWN_FRAC}"
  --window_min "${WINDOW_MIN}"
  --window_max "${WINDOW_MAX}"
  --window_warmup_steps "${WINDOW_WARMUP_STEPS}"
  --num_iterations "${NUM_ITERS}"
  --grad_clip "${GRAD_CLIP}"
  --seed "${SEED}"
  --val_every_steps "${VAL_EVERY_STEPS}"
  --val_tokens "${VAL_TOKENS}"
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}"
  --stop_mode "${STOP_MODE}"
  --dtype "${DTYPE}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
  --compile
  --tensorcores
)

if [[ "${STOP_MODE}" == "const_loss" ]]; then
  CMD+=(--loss_threshold "${LOSS_THRESHOLD}")
elif [[ -n "${STOP_EPOCH_FRAC:-}" ]]; then
  CMD+=(--stop_epoch_frac "${STOP_EPOCH_FRAC}")
fi
if [[ -n "${WINDOW_WARMUP_EPOCHS:-}" ]]; then
  CMD+=(--window_warmup_epochs "${WINDOW_WARMUP_EPOCHS}")
fi

"${CMD[@]}"
