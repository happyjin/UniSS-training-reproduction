#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {dev|eval} OUTPUT_DIR LIMIT_RECORDS_OR_0" >&2
  exit 2
fi

SPLIT="$1"
OUTPUT_DIR="$2"
LIMIT_RECORDS="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

case "${SPLIT}" in
  dev)
    GPUS="${DEV_GPUS}"
    PORT="${DEV_MASTER_PORT}"
    SAMPLES="${DEV_SAMPLES}"
    SCHEDULES="${DEV_SCHEDULES}"
    ;;
  eval)
    GPUS="${EVAL_GPUS}"
    PORT="${EVAL_MASTER_PORT}"
    SAMPLES="${EVAL_SAMPLES}"
    SCHEDULES="${EVAL_SCHEDULES}"
    ;;
  *)
    echo "Unsupported split: ${SPLIT}" >&2
    exit 2
    ;;
esac

[[ -d "${HF_EXPORT}" ]] || { echo "Missing HF export: ${HF_EXPORT}" >&2; exit 1; }
[[ -f "${SAMPLES}" && -f "${SCHEDULES}" ]] || { echo "Missing ${SPLIT} inputs" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --model "${HF_EXPORT}"
  --samples "${SAMPLES}"
  --schedules "${SCHEDULES}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --dtype "${DTYPE}"
  --attention-implementation "${ATTENTION_IMPLEMENTATION}"
  --max-batch-tokens "${MAX_BATCH_TOKENS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --logit-event-batch "${LOGIT_EVENT_BATCH}"
  --warmup-batches "${WARMUP_BATCHES}"
  --warmup-batch-size "${WARMUP_BATCH_SIZE}"
  --progress-interval 10
)
if [[ "${LIMIT_RECORDS}" != "0" ]]; then
  ARGS+=(--limit-records "${LIMIT_RECORDS}")
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES="${GPUS}" "${ENV_ROOT}/bin/torchrun" \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --master-port "${PORT}" \
  --nproc-per-node 4 \
  -m evaluation.simultaneous_streaming.stage3_action_eval \
  "${ARGS[@]}"
