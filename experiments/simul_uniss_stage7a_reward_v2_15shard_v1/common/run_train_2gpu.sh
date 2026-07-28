#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {r1|r2|r3} [--smoke]" >&2
  exit 2
fi
VARIANT="$1"
SMOKE="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"

case "${VARIANT}" in
  r1) LABEL=r1_rebalanced_coverage; GPU_LIST="${R1_GPUS}"; MASTER_PORT="${R1_MASTER_PORT}"; ADAPTIVE_KL_TARGET=0.0 ;;
  r2) LABEL=r2_explicit_latency; GPU_LIST="${R2_GPUS}"; MASTER_PORT="${R2_MASTER_PORT}"; ADAPTIVE_KL_TARGET=0.0 ;;
  r3) LABEL=r3_bilingual_adaptive; GPU_LIST="${R3_GPUS}"; MASTER_PORT="${R3_MASTER_PORT}"; ADAPTIVE_KL_TARGET="${R3_ADAPTIVE_KL_TARGET}" ;;
  *) echo "Unsupported Reward-v2 variant: ${VARIANT}" >&2; exit 2 ;;
esac

TRAIN_STEPS_LOCAL="${TRAIN_STEPS}"
SFT_WARMUP_LOCAL="${SFT_WARMUP_STEPS}"
MAX_BATCH_TOKENS_LOCAL="${MAX_BATCH_TOKENS_PER_GPU}"
MAX_BATCH_SIZE_LOCAL="${MAX_BATCH_SIZE_PER_GPU}"
VALIDATION_RECORDS_LOCAL="${VALIDATION_RECORDS}"
SHUFFLE_BUFFER_LOCAL="${SHUFFLE_BUFFER_SIZE}"
RUN_SUFFIX="${FORMAL_RUN_SUFFIX:-full}"
if [[ "${SMOKE}" == "--smoke" ]]; then
  TRAIN_STEPS_LOCAL=3
  SFT_WARMUP_LOCAL=1
  MAX_BATCH_TOKENS_LOCAL=32768
  MAX_BATCH_SIZE_LOCAL=64
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

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 2 > "${LOG_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${TRAIN_ENV}/bin/torchrun" \
  --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
  --master-port "${MASTER_PORT}" --nproc-per-node 2 \
  -m training.simul_uniss.stage7a.train \
  --mode grpo --reward-version "${VARIANT}" \
  --model "${STAGE6_MODEL}" --train-samples "${TRAIN_SAMPLES}" \
  --valid-samples "${DEV_SAMPLES}" --output-dir "${OUTPUT_DIR}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" --device cuda --dtype bf16 \
  --attention-implementation flash_attention_2 \
  --train-steps "${TRAIN_STEPS_LOCAL}" --sft-warmup-steps "${SFT_WARMUP_LOCAL}" \
  --group-size "${GROUP_SIZE}" --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" --warmup-steps "${LR_WARMUP_STEPS}" \
  --kl-beta "${KL_BETA}" --adaptive-kl-target "${ADAPTIVE_KL_TARGET}" \
  --sft-replay-weight "${SFT_REPLAY_WEIGHT}" --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS_LOCAL}" --max-batch-size "${MAX_BATCH_SIZE_LOCAL}" \
  --eval-max-batch-tokens "${EVAL_MAX_BATCH_TOKENS_PER_GPU}" \
  --eval-max-batch-size "${EVAL_MAX_BATCH_SIZE_PER_GPU}" \
  --shuffle-buffer-size "${SHUFFLE_BUFFER_LOCAL}" --validation-records "${VALIDATION_RECORDS_LOCAL}" \
  --log-interval "$([[ "${SMOKE}" == "--smoke" ]] && echo 1 || echo "${LOG_INTERVAL}")" \
  --eval-interval "$([[ "${SMOKE}" == "--smoke" ]] && echo 1 || echo "${EVAL_INTERVAL}")" \
  --save-interval "$([[ "${SMOKE}" == "--smoke" ]] && echo 1 || echo "${SAVE_INTERVAL}")" \
  --seed "${BASE_SEED}" 2>&1 | tee "${LOG_DIR}/train.log"

cleanup; trap - EXIT
if [[ "${SMOKE}" != "--smoke" ]]; then
  EXPORT_DIR="${EXPORT_ROOT}/${LABEL}_best_hf"
  POST_LOG="${LOG_DIR}/post_training.log"
  {
    mkdir -p "${EXPORT_ROOT}"
    # Concurrent policy exports can contend while materializing the shared base
    # checkpoint/tokenizer. Serialize only the short export step; dev evaluation
    # starts immediately after the lock is released on the run's own two GPUs.
    (
      flock -x 9
      if [[ ! -f "${EXPORT_DIR}/EXPORT_COMPLETE" ]]; then
        [[ ! -e "${EXPORT_DIR}" ]] || {
          echo "Refusing incomplete export directory: ${EXPORT_DIR}" >&2
          exit 1
        }
        "${TRAIN_ENV}/bin/python" -m training.simul_uniss.stage7a.export_policy_model \
          --checkpoint "${OUTPUT_DIR}/best.pt" --output-dir "${EXPORT_DIR}"
      fi
    ) 9>"${EXPORT_ROOT}/.policy_export.lock"
    "${ROOT}/evaluation/run_dev_2gpu.sh" "${LABEL}" "${EXPORT_DIR}" "${GPU_LIST}" 0.0
  } 2>&1 | tee "${POST_LOG}"
fi
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "LOG_DIR=${LOG_DIR}"
