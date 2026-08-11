#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"
MANIFEST="${PACKED_ROOT}/epoch_manifest.json"
REPLAY_ROOT="${REPO_ROOT}/data/megatron/uniss_true_subsecond_pilot15_v1"
PHASE3_FINGERPRINT="${REPO_ROOT}/data/processed/uniss_phase3_true_subsecond_deadline_full198_v1/model_handoff/phase3_embedding_fingerprint.json"
VALID_REPLAY_PACKED="${REPO_ROOT}/data/megatron/validation_unist_dev/phase3_valid_packed.jsonl"
VALID_REPLAY_OFFSETS="${REPO_ROOT}/data/processed/uniss_phase3_true_subsecond_deadline_full198_v1/validation/phase3_valid.u64"

for required in "${MANIFEST}" "${PHASE3_FINGERPRINT}" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt" \
  "${VALID_REPLAY_PACKED}" "${VALID_REPLAY_OFFSETS}"; do
  [[ -f "${required}" ]] || { echo "missing training input: ${required}" >&2; exit 2; }
done

mapfile -t geometry < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for key in ('train_iters','warmup_iters','action_write_weight','safe_positive_alpha'):
    print(value[key])
PY
)
TRAIN_ITERS="${geometry[0]}"
WARMUP_ITERS="${geometry[1]}"
ACTION_WRITE_WEIGHT="${geometry[2]}"
SAFE_POSITIVE_ALPHA="${geometry[3]}"
HANDOFF_ITER=$(( TRAIN_ITERS < 15 ? TRAIN_ITERS : 15 ))

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${REPORT_ROOT}" "${SAVE_ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TB_SESSION="uniss_true_subsecond_pilot15_v2_tensorboard"
if ! tmux has-session -t "${TB_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${TB_SESSION}" \
    "cd ${REPO_ROOT} && bash ${SCRIPT_DIR}/start_tensorboard.sh"
fi

TELEMETRY="${LOG_ROOT}/train_gpu_telemetry.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,power.limit \
  --format=csv -l 5 > "${TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT INT TERM

common=(
  "RUN_NAME=${EXPERIMENT_NAME}"
  "RUN_SAVE_DIR=${SAVE_ROOT}"
  "RUN_TB_DIR=${RUN_ROOT}/tensorboard"
  "RUN_LOG=${LOG_ROOT}/train.log"
  "RUN_TRAJECTORY_PACKED=${PACKED_ROOT}/packed_trajectory.jsonl"
  "RUN_TRAJECTORY_OFFSETS=${PACKED_ROOT}/packed_trajectory.offsets.u64"
  "RUN_REPLAY_PACKED=${REPLAY_ROOT}/packed_replay.jsonl"
  "RUN_REPLAY_OFFSETS=${PACKED_ROOT}/replay_subset.offsets.u64"
  "RUN_VALID_REPLAY_PACKED=${VALID_REPLAY_PACKED}"
  "RUN_VALID_REPLAY_OFFSETS=${VALID_REPLAY_OFFSETS}"
  "RUN_TRAIN_ITERS=${TRAIN_ITERS}"
  "RUN_WARMUP_ITERS=${WARMUP_ITERS}"
  RUN_NPROC=8 RUN_MBS=2 RUN_GBS=128 RUN_SEQ_LENGTH=18000
  RUN_SAVE_INTERVAL=15 RUN_EVAL_INTERVAL=15 RUN_LOG_INTERVAL=1
  RUN_MASTER_PORT=29721 RUN_SMOKE=0 RUN_AUDIT_GRADIENTS=0
  RUN_FULL_VALIDATION=0 NUM_WORKERS=8
)
extra=(
  --true-action-write-weight "${ACTION_WRITE_WEIGHT}"
  --true-safe-positive-alpha "${SAFE_POSITIVE_ALPHA}"
)

tracker=-1
[[ -s "${SAVE_ROOT}/latest_checkpointed_iteration.txt" ]] && \
  tracker="$(tr -d '[:space:]' < "${SAVE_ROOT}/latest_checkpointed_iteration.txt")"
if [[ -f "${RUN_ROOT}/EPOCH_COMPLETE" && "${tracker}" == "${TRAIN_ITERS}" ]]; then
  echo "${EXPERIMENT_NAME} already completed one trajectory epoch"
  exit 0
fi

if (( tracker < 0 )); then
  if find "${SAVE_ROOT}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing untracked non-empty checkpoint path: ${SAVE_ROOT}" >&2
    exit 2
  fi
  env "${common[@]}" RUN_LOAD="${PHASE3_NATIVE_CHECKPOINT}" \
    RUN_FINETUNE=1 RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 \
    RUN_STRICTNESS=log_all RUN_EXIT_INTERVAL="${HANDOFF_ITER}" \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" \
    "${extra[@]}"
  tracker="$(tr -d '[:space:]' < "${SAVE_ROOT}/latest_checkpointed_iteration.txt")"
  [[ "${tracker}" == "${HANDOFF_ITER}" ]] || {
    echo "expected handoff checkpoint ${HANDOFF_ITER}, got ${tracker}" >&2
    exit 1
  }
fi

if (( tracker < TRAIN_ITERS )); then
  env "${common[@]}" RUN_LOAD="${SAVE_ROOT}" \
    RUN_FINETUNE=0 RUN_LOAD_OPTIM=1 RUN_LOAD_RNG=1 \
    RUN_STRICTNESS=raise_all \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" \
    "${extra[@]}"
fi

tracker="$(tr -d '[:space:]' < "${SAVE_ROOT}/latest_checkpointed_iteration.txt")"
[[ "${tracker}" == "${TRAIN_ITERS}" ]] || {
  echo "training stopped at ${tracker}, expected ${TRAIN_ITERS}" >&2
  exit 1
}
if rg -q 'grad norm: (nan|inf|-inf)|number of (skipped|nan) iterations: +[1-9]' "${LOG_ROOT}/train.log"; then
  echo "non-finite or skipped iteration found in training log" >&2
  exit 1
fi
printf 'completed_at=%s\niteration=%s\nresume_verified_from=%s\n' \
  "$(date -u +%FT%TZ)" "${TRAIN_ITERS}" "${HANDOFF_ITER}" > "${RUN_ROOT}/EPOCH_COMPLETE"
