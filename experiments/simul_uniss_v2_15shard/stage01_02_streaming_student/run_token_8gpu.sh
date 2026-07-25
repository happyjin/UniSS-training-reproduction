#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

cmd=(torchrun
  --nproc_per_node "${SIMUL_NPROC_PER_NODE}"
  --master_addr "${SIMUL_MASTER_ADDR}"
  --master_port "${STAGE1_TOKEN_MASTER_PORT}"
  -m training.simul_uniss.train_streaming_student
  --schedules "${SCHEDULES_JSONL}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --output-dir "${STAGE1_TOKEN_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE1_TOKEN_TENSORBOARD_DIR}"
  --qwen-checkpoint-root "${QWEN_CHECKPOINT_ROOT}"
  --device cuda
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
  --seed "${SIMUL_DATA_SEED}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

[[ -f "${POLICY_TOKENIZER_MODEL}" ]] || { echo "Missing policy tokenizer: ${POLICY_TOKENIZER_MODEL}" >&2; exit 1; }
[[ -f "${SCHEDULES_JSONL}" ]] || { echo "Missing schedules: ${SCHEDULES_JSONL}" >&2; exit 1; }
[[ ! -e "${STAGE1_TOKEN_OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${STAGE1_TOKEN_OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}" "${STAGE1_TOKEN_TENSORBOARD_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage01_02_streaming_token_student.log"
