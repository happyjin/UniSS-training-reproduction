#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/config.env"

for value in \
  "${TRAJECTORY_PACKED}" "${TRAJECTORY_PACKED}.offsets.bin" \
  "${TRAINING_MANIFEST}" "${REPLAY_SUBSET_OFFSETS}" \
  "${VALID_TRAJECTORY_PACKED}" "${VALID_TRAJECTORY_PACKED}.offsets.bin" \
  "${VALID_REPLAY_PACKED}" "${VALID_REPLAY_OFFSETS}" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${value}" ]] || { echo "Missing overfit2 artifact: ${value}" >&2; exit 1; }
done

read -r TRAIN_ITERS WARMUP_ITERS MANIFEST_EPOCHS < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "complete" or not value.get("strict_global_shuffle"):
    raise SystemExit("overfit2 training manifest is incomplete")
print(value["train_iters"], value["warmup_iters"], value["coverage_epochs"])
PY
)
[[ "${MANIFEST_EPOCHS}" == "${COVERAGE_EPOCHS}" ]] || {
  echo "Coverage epochs differ: ${MANIFEST_EPOCHS} != ${COVERAGE_EPOCHS}" >&2
  exit 1
}

if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  current="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
  if (( current == TRAIN_ITERS )); then
    echo "runtime-parity overfit2 already complete at ${TRAIN_ITERS}"
    exit 0
  fi
  LOAD="${SAVE_DIR}"
  FINETUNE=0
  LOAD_OPTIM=1
  LOAD_RNG=1
  STRICTNESS=raise_all
else
  if [[ -e "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty overfit2 checkpoint directory: ${SAVE_DIR}" >&2
    exit 1
  fi
  LOAD="${PHASE3_NATIVE_CHECKPOINT}"
  FINETUNE=1
  LOAD_OPTIM=0
  LOAD_RNG=0
  STRICTNESS=log_all
fi

export CUDA_VISIBLE_DEVICES COVERAGE_EPOCHS REPLAY_FRACTION
export LR_QWEN_LORA LR_FRONTEND LR_NEW_HEADS MIN_LR
INNER_ARGS=()
[[ "${DRY_RUN}" == "1" ]] && INNER_ARGS+=(--dry-run)
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_SAVE_DIR="${SAVE_DIR}" RUN_TB_DIR="${TB_DIR}" RUN_LOG="${LOG_PATH}" \
RUN_TRAINING_MANIFEST="${TRAINING_MANIFEST}" \
RUN_TRAJECTORY_PACKED="${TRAJECTORY_PACKED}" \
RUN_TRAJECTORY_OFFSETS="${TRAJECTORY_PACKED}.offsets.bin" \
RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" \
RUN_REPLAY_OFFSETS="${REPLAY_SUBSET_OFFSETS}" \
RUN_VALID_TRAJECTORY_PACKED="${VALID_TRAJECTORY_PACKED}" \
RUN_VALID_TRAJECTORY_OFFSETS="${VALID_TRAJECTORY_PACKED}.offsets.bin" \
RUN_VALID_REPLAY_PACKED="${VALID_REPLAY_PACKED}" \
RUN_VALID_REPLAY_OFFSETS="${VALID_REPLAY_OFFSETS}" \
RUN_ENTRYPOINT="${RUN_ENTRYPOINT:-${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/pretrain_overfit2.py}" \
RUN_LORA_DROPOUT=0 RUN_ATTENTION_DROPOUT=0 RUN_HIDDEN_DROPOUT=0 \
RUN_FULL_VALIDATION=1 RUN_EVAL_INTERVAL=50 EVAL_ITERS=1 \
RUN_TRAIN_ITERS="${TRAIN_ITERS}" RUN_WARMUP_ITERS="${WARMUP_ITERS}" \
RUN_NPROC=8 RUN_MBS="${MICRO_BATCH_SIZE}" RUN_GBS="${GLOBAL_BATCH_SIZE}" \
RUN_SEQ_LENGTH="${SEQ_LENGTH}" RUN_MASTER_PORT="${MASTER_PORT}" \
RUN_LOAD="${LOAD}" RUN_FINETUNE="${FINETUNE}" \
RUN_LOAD_OPTIM="${LOAD_OPTIM}" RUN_LOAD_RNG="${LOAD_RNG}" \
RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE=0 RUN_AUDIT_GRADIENTS=1 \
RUN_SAVE_INTERVAL=50 RUN_LOG_INTERVAL=1 \
bash "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh" \
  "${INNER_ARGS[@]}" "$@"

[[ "${DRY_RUN}" == "1" ]] && exit 0
actual="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
[[ "${actual}" == "${TRAIN_ITERS}" ]] || {
  echo "runtime-parity overfit2 stopped at ${actual}, expected ${TRAIN_ITERS}" >&2
  exit 1
}
