#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SMOKE=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

if [[ ! -f "${SCHEDULES_JSONL}" ]]; then
  echo "Missing schedules: ${SCHEDULES_JSONL}; run prepare_bootstrap_15shard.sh first" >&2
  exit 1
fi
if [[ ! -f "${QWEN_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing checkpoint pointer: ${QWEN_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" >&2
  exit 1
fi

mkdir -p "${POLICY_TOKENIZER_DIR}" "${STAGE1_OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"
shards=()
for ((index=SHARD_START; index<SHARD_START+SHARD_COUNT; index++)); do
  printf -v shard '%s/train-%05d.parquet' "${UNIST_ROOT}" "${index}"
  shards+=("${shard}")
done

tokenizer_cmd=(python -m training.simul_uniss.policy_tokenizer
  --input "${shards[@]}"
  --output-dir "${POLICY_TOKENIZER_DIR}"
  --vocab-size "${POLICY_VOCAB_SIZE}"
  --num-threads 8
)

train_cmd=(python -m training.simul_uniss.train_streaming_student
  --schedules "${SCHEDULES_JSONL}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --output-dir "${STAGE1_OUTPUT_DIR}"
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --qwen-checkpoint-root "${QWEN_CHECKPOINT_ROOT}"
  --device "${STAGE1_DEVICE}"
  --batch-size "${STAGE1_BATCH_SIZE}"
  --max-steps "${STAGE1_MAX_STEPS}"
  --learning-rate "${STAGE1_LEARNING_RATE}"
  --hidden-size "${STAGE1_HIDDEN_SIZE}"
  --num-layers "${STAGE1_NUM_LAYERS}"
  --num-heads "${STAGE1_NUM_HEADS}"
  --max-source-tokens "${STAGE1_MAX_SOURCE_TOKENS}"
  --validation-records "${STAGE1_VALIDATION_RECORDS}"
  --eval-interval "${STAGE1_EVAL_INTERVAL}"
  --save-interval "${STAGE1_SAVE_INTERVAL}"
)

if [[ "${SMOKE}" == "1" ]]; then
  tokenizer_cmd+=(--limit-records 1000)
  train_cmd=(python -m training.simul_uniss.train_streaming_student
    --schedules "${SCHEDULES_JSONL}"
    --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
    --output-dir "${STAGE1_OUTPUT_DIR}/smoke"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --qwen-checkpoint-root "${QWEN_CHECKPOINT_ROOT}"
    --device cpu
    --batch-size 1
    --max-steps 2
    --hidden-size 32
    --num-layers 1
    --num-heads 4
    --max-source-tokens 64
    --validation-records 4
    --eval-interval 1
    --eval-batches 1
    --log-interval 1
    --save-interval 1
  )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ ! -f "${POLICY_TOKENIZER_MODEL}" ]]; then
    printf '%q ' "${tokenizer_cmd[@]}"; printf '\n'
  fi
  printf '%q ' "${train_cmd[@]}"; printf '\n'
  exit 0
fi

{
  if [[ ! -f "${POLICY_TOKENIZER_MODEL}" ]]; then
    "${tokenizer_cmd[@]}"
  fi
  "${train_cmd[@]}"
} 2>&1 | tee -a "${LOG_DIR}/stage1_bootstrap_student.log"
