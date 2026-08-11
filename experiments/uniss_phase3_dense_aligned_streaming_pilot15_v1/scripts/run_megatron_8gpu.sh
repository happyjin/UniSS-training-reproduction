#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"

for value in \
  "${TRAINING_MANIFEST}" "${SHUFFLE_AUDIT}" \
  "${TRAJECTORY_PACKED}" "${TRAJECTORY_OFFSETS}" \
  "${PHASE3_REPLAY_PACKED}" "${REPLAY_SUBSET_OFFSETS}" \
  "${VALID_TRAJECTORY_PACKED}" "${VALID_TRAJECTORY_OFFSETS}" \
  "${VALID_REPLAY_PACKED}" "${VALID_REPLAY_OFFSETS}"; do
  [[ -f "${value}" ]] || { echo "Missing formal artifact: ${value}" >&2; exit 1; }
done

read -r TRAIN_ITERS WARMUP_ITERS TOTAL_SAMPLES EPOCH_SAMPLES TRAJECTORY_COUNT REPLAY_COUNT < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "complete" or not value.get("strict_global_shuffle"):
    raise SystemExit("training manifest is not complete strict-global-shuffle geometry")
print(
    value["train_iters"], value["warmup_iters"], value["total_samples"],
    value["epoch_samples"], value["trajectory_count"], value["replay_selected"]
)
PY
)

TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
current=-1
[[ -s "${TRACKER}" ]] && current="$(tr -d '[:space:]' < "${TRACKER}")"
if (( current == TRAIN_ITERS )); then
  echo "dense-aligned formal run already complete at ${TRAIN_ITERS}"
  exit 0
fi
if (( current > TRAIN_ITERS )); then
  echo "checkpoint ${current} exceeds computed target ${TRAIN_ITERS}" >&2
  exit 1
fi
if (( current < 0 )); then
  if [[ -e "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing fresh run in non-empty untracked directory: ${SAVE_DIR}" >&2
    exit 1
  fi
  LOAD="${PHASE3_NATIVE_CHECKPOINT}"
  FINETUNE=1
  LOAD_OPTIM=0
  LOAD_RNG=0
  STRICTNESS=log_all
else
  LOAD="${SAVE_DIR}"
  FINETUNE=0
  LOAD_OPTIM=1
  LOAD_RNG=1
  STRICTNESS=raise_all
fi

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/manifest.txt" <<EOF
experiment=${EXPERIMENT_NAME}
updated_at=$(date -u +%FT%TZ)
repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
phase3_native_checkpoint=${PHASE3_NATIVE_CHECKPOINT}
trajectory_packed=${TRAJECTORY_PACKED}
trajectory_count=${TRAJECTORY_COUNT}
replay_packed=${PHASE3_REPLAY_PACKED}
replay_subset_offsets=${REPLAY_SUBSET_OFFSETS}
replay_count=${REPLAY_COUNT}
coverage_epochs=${COVERAGE_EPOCHS}
epoch_samples=${EPOCH_SAMPLES}
total_samples=${TOTAL_SAMPLES}
train_iters=${TRAIN_ITERS}
warmup_iters=${WARMUP_ITERS}
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
sequence_length=${SEQ_LENGTH}
shuffle=independent_full_randperm_over_every_complete_18k_pack_per_coverage_epoch
shuffle_seed=${SEED}
session_internal_event_order=preserved
tensorboard_dir=${TB_DIR}
tensorboard_port=${TENSORBOARD_PORT}
valid_trajectory_packed=${VALID_TRAJECTORY_PACKED}
valid_replay_packed=${VALID_REPLAY_PACKED}
EOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
RUN_NAME="${EXPERIMENT_NAME}" RUN_SAVE_DIR="${SAVE_DIR}" \
RUN_TB_DIR="${TB_DIR}" RUN_LOG="${LOG_PATH}" \
RUN_TRAINING_MANIFEST="${TRAINING_MANIFEST}" \
RUN_TRAJECTORY_PACKED="${TRAJECTORY_PACKED}" \
RUN_TRAJECTORY_OFFSETS="${TRAJECTORY_OFFSETS}" \
RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" \
RUN_REPLAY_OFFSETS="${REPLAY_SUBSET_OFFSETS}" \
RUN_VALID_TRAJECTORY_PACKED="${VALID_TRAJECTORY_PACKED}" \
RUN_VALID_TRAJECTORY_OFFSETS="${VALID_TRAJECTORY_OFFSETS}" \
RUN_VALID_REPLAY_PACKED="${VALID_REPLAY_PACKED}" \
RUN_VALID_REPLAY_OFFSETS="${VALID_REPLAY_OFFSETS}" \
RUN_FULL_VALIDATION=1 RUN_TRAIN_ITERS="${TRAIN_ITERS}" \
RUN_WARMUP_ITERS="${WARMUP_ITERS}" RUN_NPROC="${NPROC_PER_NODE}" \
RUN_MBS="${MICRO_BATCH_SIZE}" RUN_GBS="${GLOBAL_BATCH_SIZE}" \
RUN_MASTER_PORT="${MASTER_PORT}" RUN_LOAD="${LOAD}" \
RUN_FINETUNE="${FINETUNE}" RUN_LOAD_OPTIM="${LOAD_OPTIM}" \
RUN_LOAD_RNG="${LOAD_RNG}" RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE=0 \
bash "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh" "$@"

[[ "${DRY_RUN}" == "1" ]] && exit 0

actual="$(tr -d '[:space:]' < "${TRACKER}")"
[[ "${actual}" == "${TRAIN_ITERS}" ]] || {
  echo "Dense-aligned formal run stopped at ${actual}, expected ${TRAIN_ITERS}" >&2
  exit 1
}
printf 'completed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${actual}" > "${RUN_DIR}/TRAINING_COMPLETE"
