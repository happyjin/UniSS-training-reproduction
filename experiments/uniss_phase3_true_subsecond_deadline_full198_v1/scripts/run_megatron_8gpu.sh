#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

for value in \
  "${TRAJECTORY_PACKED}" "${TRAJECTORY_PACKED}.count" "${TRAJECTORY_OFFSETS}" \
  "${PHASE3_REPLAY_PACKED}" "${PHASE3_REPLAY_OFFSETS}" \
  "${DEV_TRAJECTORY_PACKED}" "${DEV_TRAJECTORY_OFFSETS}" \
  "${VALID_REPLAY_PACKED}" "${VALID_REPLAY_OFFSETS}"; do
  [[ -f "${value}" ]] || { echo "Missing formal training artifact: ${value}" >&2; exit 1; }
done
read -r TRAIN_ITERS SCHEDULE_COUNT WARMUP_ITERS REPLAY_COUNT TRAJECTORY_COUNT < <(
  "${PYTHON}" - "${PHASE3_REPLAY_OFFSETS}.json" "${TRAJECTORY_PACKED}.count" <<'PY'
import json, sys
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.packed_epoch import JointPackedEpochGeometry
replay = int(json.load(open(sys.argv[1]))["records"])
trajectory = int(open(sys.argv[2]).read().strip())
geometry = JointPackedEpochGeometry(replay, trajectory, 16, 128)
print(geometry.train_iters, geometry.schedule_count, geometry.warmup_iters, replay, trajectory)
PY
)
TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
current=-1
[[ -s "${TRACKER}" ]] && current="$(tr -d '[:space:]' < "${TRACKER}")"
if (( current == TRAIN_ITERS )); then
  echo "formal run already complete at ${TRAIN_ITERS}"
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
created_at=$(date -u +%FT%TZ)
repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
phase3_native_checkpoint=${PHASE3_NATIVE_CHECKPOINT}
trajectory_packed=${TRAJECTORY_PACKED}
trajectory_count=${TRAJECTORY_COUNT}
replay_packed=${PHASE3_REPLAY_PACKED}
replay_count=${REPLAY_COUNT}
schedule_count=${SCHEDULE_COUNT}
train_iters=${TRAIN_ITERS}
warmup_iters=${WARMUP_ITERS}
micro_batch_size=2
global_batch_size=128
sequence_length=18000
shuffle=phase3_v4_equivalent_global_source_group_permutation_then_curriculum_phase_shuffle
shuffle_seed=${SEED}
tensorboard_dir=${TB_DIR}
valid_trajectory_packed=${DEV_TRAJECTORY_PACKED}
valid_replay_packed=${VALID_REPLAY_PACKED}
EOF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RUN_NAME="${EXPERIMENT_NAME}" RUN_SAVE_DIR="${SAVE_DIR}" \
RUN_TB_DIR="${TB_DIR}" RUN_LOG="${LOG_PATH}" \
RUN_TRAJECTORY_PACKED="${TRAJECTORY_PACKED}" \
RUN_TRAJECTORY_OFFSETS="${TRAJECTORY_OFFSETS}" \
RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" \
RUN_REPLAY_OFFSETS="${PHASE3_REPLAY_OFFSETS}" \
RUN_VALID_TRAJECTORY_PACKED="${DEV_TRAJECTORY_PACKED}" \
RUN_VALID_TRAJECTORY_OFFSETS="${DEV_TRAJECTORY_OFFSETS}" \
RUN_VALID_REPLAY_PACKED="${VALID_REPLAY_PACKED}" \
RUN_VALID_REPLAY_OFFSETS="${VALID_REPLAY_OFFSETS}" \
RUN_FULL_VALIDATION=1 RUN_TRAIN_ITERS="${TRAIN_ITERS}" \
RUN_WARMUP_ITERS="${WARMUP_ITERS}" RUN_NPROC=8 RUN_MBS=2 RUN_GBS=128 \
RUN_MASTER_PORT="${MASTER_PORT}" RUN_LOAD="${LOAD}" \
RUN_FINETUNE="${FINETUNE}" RUN_LOAD_OPTIM="${LOAD_OPTIM}" RUN_LOAD_RNG="${LOAD_RNG}" \
RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE=0 \
bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
actual="$(tr -d '[:space:]' < "${TRACKER}")"
[[ "${actual}" == "${TRAIN_ITERS}" ]] || { echo "formal run stopped at ${actual}" >&2; exit 1; }
printf 'completed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${actual}" > "${RUN_DIR}/TRAINING_COMPLETE"
echo "formal full198 true-subsecond run complete at iteration ${actual}"
