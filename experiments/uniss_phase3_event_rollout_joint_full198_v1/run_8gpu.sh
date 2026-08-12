#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config_canary.env"

for value in \
  "${TRAJECTORY_PACKED}" "${TRAJECTORY_PACKED}.offsets.bin" \
  "${VALID_TRAJECTORY_PACKED}" "${VALID_TRAJECTORY_PACKED}.offsets.bin" \
  "${TRAINING_MANIFEST}" "${REPLAY_SUBSET_OFFSETS}" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${value}" ]] || { echo "Missing event-rollout artifact: ${value}" >&2; exit 1; }
done
[[ "$(tr -d '[:space:]' < "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt")" == "9075" ]] || {
  echo "Phase3 root is not pinned to iter_0009075" >&2
  exit 1
}

read -r TRAIN_ITERS MANIFEST_WARMUP_ITERS MANIFEST_EPOCHS < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "complete" or not value.get("strict_global_shuffle"):
    raise SystemExit("event-rollout manifest is incomplete")
if value.get("session_internal_event_order") != "preserved":
    raise SystemExit("event order is not preserved")
print(value["train_iters"], value["warmup_iters"], value["coverage_epochs"])
PY
)
WARMUP_ITERS="${CANARY_WARMUP_ITERS:-${MANIFEST_WARMUP_ITERS}}"
[[ "${MANIFEST_EPOCHS}" == "${COVERAGE_EPOCHS}" ]] || {
  echo "Coverage epochs differ from frozen manifest" >&2
  exit 1
}

TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
if [[ -s "${TRACKER}" ]]; then
  current="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( current == TRAIN_ITERS )); then
    echo "event-rollout run already complete at ${TRAIN_ITERS}"
    exit 0
  fi
  LOAD="${SAVE_DIR}"
  FINETUNE=0
  LOAD_OPTIM=1
  LOAD_RNG=1
  STRICTNESS=raise_all
else
  if [[ -e "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty event-rollout checkpoint directory" >&2
    exit 1
  fi
  LOAD="${PHASE3_NATIVE_CHECKPOINT}"
  FINETUNE=1
  LOAD_OPTIM=0
  LOAD_RNG=0
  STRICTNESS=log_all
fi

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/manifest.txt" <<EOF
experiment=${EXPERIMENT_NAME}
created_at=$(date -u +%FT%TZ)
repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
base_checkpoint=${PHASE3_NATIVE_CHECKPOINT}/iter_0009075
training=one_continuous_event_rollout_joint_sft
runtime=v9_fused_commit_plus_v12_four_unit_microblocks
rollout=exact_variable_wait_write_text_semantic_continuation_eos_persistent_kv
recovery=oracle_supervision_on_model_generated_history
shuffle=full_randperm_over_complete_packs_session_event_order_preserved
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
sequence_length=${SEQ_LENGTH}
coverage_epochs=${COVERAGE_EPOCHS}
train_iters=${TRAIN_ITERS}
tensorboard_dir=${TB_DIR}
tensorboard_port=${TENSORBOARD_PORT}
EOF

export CUDA_VISIBLE_DEVICES COVERAGE_EPOCHS REPLAY_FRACTION
export LR_QWEN_LORA LR_FRONTEND LR_NEW_HEADS MIN_LR WEIGHT_DECAY
export NUM_WORKERS NPZ_LRU_CAPACITY SEED
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
RUN_VALID_REPLAY_PACKED="" RUN_VALID_REPLAY_OFFSETS="" \
RUN_ENTRYPOINT="${HERE}/training/pretrain_event_rollout_megatron.py" \
RUN_LORA_DROPOUT=0 RUN_ATTENTION_DROPOUT=0 RUN_HIDDEN_DROPOUT=0 \
RUN_FULL_VALIDATION=1 RUN_EVAL_INTERVAL="${EVAL_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}" \
RUN_TRAIN_ITERS="${TRAIN_ITERS}" RUN_WARMUP_ITERS="${WARMUP_ITERS}" \
RUN_NPROC=8 RUN_MBS="${MICRO_BATCH_SIZE}" RUN_GBS="${GLOBAL_BATCH_SIZE}" \
RUN_SEQ_LENGTH="${SEQ_LENGTH}" RUN_MASTER_PORT="${MASTER_PORT}" \
RUN_LOAD="${LOAD}" RUN_FINETUNE="${FINETUNE}" \
RUN_LOAD_OPTIM="${LOAD_OPTIM}" RUN_LOAD_RNG="${LOAD_RNG}" \
RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE=0 RUN_AUDIT_GRADIENTS=1 \
RUN_SAVE_INTERVAL="${SAVE_INTERVAL}" RUN_LOG_INTERVAL="${LOG_INTERVAL}" \
bash "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh" \
  "${INNER_ARGS[@]}" "$@"
