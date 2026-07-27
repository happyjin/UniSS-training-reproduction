#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {e1|e2|e3} [--smoke]" >&2
  exit 2
fi

EXPERIMENT="$1"
SMOKE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

case "${EXPERIMENT}" in
  e1)
    MODE=sft
    GROUP_SIZE=4
    GPU_LIST="${E1_GPUS}"
    MASTER_PORT="${E1_MASTER_PORT}"
    LABEL=e1_continued_sft
    ;;
  e2)
    MODE=grpo
    GROUP_SIZE=4
    GPU_LIST="${E2_GPUS}"
    MASTER_PORT="${E2_MASTER_PORT}"
    LABEL=e2_grpo_g4
    ;;
  e3)
    MODE=grpo
    GROUP_SIZE=8
    GPU_LIST="${E3_GPUS}"
    MASTER_PORT="${E3_MASTER_PORT}"
    LABEL=e3_grpo_g8
    ;;
  *)
    echo "Unsupported experiment: ${EXPERIMENT}" >&2
    exit 2
    ;;
esac

TRAIN_STEPS_LOCAL="${TRAIN_STEPS}"
SFT_WARMUP_LOCAL="${SFT_WARMUP_STEPS}"
MAX_BATCH_TOKENS_LOCAL="${MAX_BATCH_TOKENS_PER_GPU}"
MAX_BATCH_SIZE_LOCAL="${MAX_BATCH_SIZE_PER_GPU}"
VALIDATION_RECORDS_LOCAL="${VALIDATION_RECORDS}"
SHUFFLE_BUFFER_LOCAL="${SHUFFLE_BUFFER_SIZE}"
RUN_SUFFIX=full
if [[ "${SMOKE}" == "--smoke" ]]; then
  TRAIN_STEPS_LOCAL=3
  SFT_WARMUP_LOCAL=1
  MAX_BATCH_TOKENS_LOCAL=16384
  MAX_BATCH_SIZE_LOCAL=32
  VALIDATION_RECORDS_LOCAL=32
  SHUFFLE_BUFFER_LOCAL=64
  RUN_SUFFIX="smoke_$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ -n "${SMOKE}" ]]; then
  echo "Unknown argument: ${SMOKE}" >&2
  exit 2
fi

OUTPUT_DIR="${CHECKPOINT_ROOT}/${LABEL}_${RUN_SUFFIX}"
TENSORBOARD_DIR="${RUN_ROOT}/tensorboard/${LABEL}_${RUN_SUFFIX}"
LOG_DIR="${LOG_ROOT}/${LABEL}_${RUN_SUFFIX}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

ARGS=(
  --mode "${MODE}"
  --model "${STAGE6_MODEL}"
  --train-samples "${TRAIN_SAMPLES}"
  --valid-samples "${DEV_SAMPLES}"
  --output-dir "${OUTPUT_DIR}"
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --device cuda
  --dtype bf16
  --attention-implementation flash_attention_2
  --train-steps "${TRAIN_STEPS_LOCAL}"
  --sft-warmup-steps "${SFT_WARMUP_LOCAL}"
  --group-size "${GROUP_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-steps "${LR_WARMUP_STEPS}"
  --kl-beta "${KL_BETA}"
  --sft-replay-weight "${SFT_REPLAY_WEIGHT}"
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"
  --max-batch-tokens "${MAX_BATCH_TOKENS_LOCAL}"
  --max-batch-size "${MAX_BATCH_SIZE_LOCAL}"
  --eval-max-batch-tokens "${EVAL_MAX_BATCH_TOKENS_PER_GPU}"
  --eval-max-batch-size "${EVAL_MAX_BATCH_SIZE_PER_GPU}"
  --shuffle-buffer-size "${SHUFFLE_BUFFER_LOCAL}"
  --validation-records "${VALIDATION_RECORDS_LOCAL}"
  --log-interval "${LOG_INTERVAL}"
  --eval-interval "$([[ "${SMOKE}" == "--smoke" ]] && echo 1 || echo "${EVAL_INTERVAL}")"
  --save-interval "$([[ "${SMOKE}" == "--smoke" ]] && echo 1 || echo "${SAVE_INTERVAL}")"
  --seed "${BASE_SEED}"
)

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 2 > "${LOG_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${TRAIN_ENV}/bin/torchrun" \
  --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
  --master-port "${MASTER_PORT}" --nproc-per-node 2 \
  -m training.simul_uniss.stage7a.train "${ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/train.log"

cleanup
trap - EXIT
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "LOG_DIR=${LOG_DIR}"
