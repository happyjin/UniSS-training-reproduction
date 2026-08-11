#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/generalize12_microblock/config.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

for value in \
  "${TRAJECTORY_PACKED}" "${TRAJECTORY_PACKED}.offsets.bin" \
  "${TRAINING_MANIFEST}" "${REPLAY_SUBSET_OFFSETS}" \
  "${VALID_TRAJECTORY_PACKED}" "${VALID_TRAJECTORY_PACKED}.offsets.bin" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${value}" ]] || { echo "Missing generalize12 artifact: ${value}" >&2; exit 1; }
done

read -r TRAIN_ITERS WARMUP_ITERS MANIFEST_EPOCHS TRAJECTORY_COUNT REPLAY_COUNT < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "complete" or not value.get("strict_global_shuffle"):
    raise SystemExit("generalize12 training manifest is incomplete")
print(value["train_iters"], value["warmup_iters"], value["coverage_epochs"],
      value["trajectory_count"], value["replay_selected"])
PY
)
[[ "${MANIFEST_EPOCHS}" == "${COVERAGE_EPOCHS}" ]] || {
  echo "Coverage epochs differ: ${MANIFEST_EPOCHS} != ${COVERAGE_EPOCHS}" >&2
  exit 1
}

TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
if [[ -s "${TRACKER}" ]]; then
  current="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( current == TRAIN_ITERS )); then
    echo "generalize12 already complete at ${TRAIN_ITERS}"
    exit 0
  fi
  if (( current > TRAIN_ITERS )); then
    echo "checkpoint ${current} exceeds target ${TRAIN_ITERS}" >&2
    exit 1
  fi
  LOAD="${SAVE_DIR}"
  FINETUNE=0
  LOAD_OPTIM=1
  LOAD_RNG=1
  STRICTNESS=raise_all
else
  if [[ -e "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty generalize12 checkpoint directory: ${SAVE_DIR}" >&2
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
updated_at=$(date -u +%FT%TZ)
repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
base_checkpoint=${PHASE3_NATIVE_CHECKPOINT}
trainable_scope=causal_microblock_semantic_head_only
microblock_size=4
semantic_classifier=frozen_phase3_embedding_rows
continuation=natural_binary_posterior
final_length=natural_1_to_4_posterior
validation=trajectory_only_because_all_replay_parameters_are_frozen
validation_dp_padding=cyclic_to_complete_8_rank_batches_strict_pcm_gate_unpadded
trajectory_packed=${TRAJECTORY_PACKED}
trajectory_count=${TRAJECTORY_COUNT}
replay_fraction=${REPLAY_FRACTION}
replay_count=${REPLAY_COUNT}
coverage_epochs=${COVERAGE_EPOCHS}
train_iters=${TRAIN_ITERS}
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
sequence_length=${SEQ_LENGTH}
shuffle=independent_full_randperm_over_every_complete_18k_pack
shuffle_seed=${SEED}
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
RUN_ENTRYPOINT="${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/generalize12_microblock/pretrain_generalize12.py" \
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

[[ "${DRY_RUN}" == "1" ]] && exit 0
actual="$(tr -d '[:space:]' < "${TRACKER}")"
[[ "${actual}" == "${TRAIN_ITERS}" ]] || {
  echo "generalize12 stopped at ${actual}, expected ${TRAIN_ITERS}" >&2
  exit 1
}
printf 'completed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${actual}" > "${RUN_DIR}/TRAINING_COMPLETE"
