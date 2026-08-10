#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

NAME="${SMOKE8_NAME:-uniss_true_subsecond_native_smoke8_50step_v1}"
SMOKE_ROOT="${DATA_ROOT}/smoke/trajectory_assembly_v2"
SAVE="${REPO_ROOT}/checkpoints/${NAME}"
RUN="${REPO_ROOT}/runs/${NAME}"
LOG="${REPO_ROOT}/logs/${NAME}.log"
mkdir -p "${RUN}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

common=(
  "RUN_NAME=${NAME}"
  "RUN_SAVE_DIR=${SAVE}"
  "RUN_TB_DIR=${RUN}/tensorboard"
  "RUN_LOG=${LOG}"
  "RUN_TRAJECTORY_PACKED=${SMOKE_ROOT}/packed_trajectory.jsonl"
  "RUN_TRAJECTORY_OFFSETS=${SMOKE_ROOT}/packed_trajectory.offsets.u64"
  "RUN_REPLAY_PACKED=${PHASE3_REPLAY_PACKED}"
  "RUN_REPLAY_OFFSETS=${REPO_ROOT}/data/processed/phase3_whisper_streamspeech_joint_v1/phase3_replay_index/smoke_first8.u64"
  RUN_TRAIN_ITERS=50 RUN_NPROC=8 RUN_MBS=2 RUN_GBS=128
  RUN_WARMUP_ITERS=5 RUN_SAVE_INTERVAL=25 RUN_LOG_INTERVAL=1
  RUN_MASTER_PORT=29712 RUN_SMOKE=1 NUM_WORKERS=0
)
tracker=-1
[[ -s "${SAVE}/latest_checkpointed_iteration.txt" ]] && \
  tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
if [[ -f "${RUN}/SMOKE_COMPLETE" && "${tracker}" == "50" ]]; then
  echo "${NAME} already passed"
  exit 0
fi
if (( tracker < 0 )); then
  [[ ! -e "${SAVE}" ]] || { echo "Refusing untracked smoke directory: ${SAVE}" >&2; exit 1; }
  env "${common[@]}" RUN_LOAD="${PHASE3_NATIVE_CHECKPOINT}" \
    RUN_FINETUNE=1 RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 \
    RUN_STRICTNESS=log_all RUN_EXIT_INTERVAL=5 \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
  tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
  [[ "${tracker}" == "5" ]] || { echo "expected interruption checkpoint 5, got ${tracker}" >&2; exit 1; }
fi
if (( tracker < 50 )); then
  env "${common[@]}" RUN_LOAD="${SAVE}" \
    RUN_FINETUNE=0 RUN_LOAD_OPTIM=1 RUN_LOAD_RNG=1 \
    RUN_STRICTNESS=raise_all \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
fi
tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
[[ "${tracker}" == "50" ]] || { echo "8-GPU smoke stopped at ${tracker}" >&2; exit 1; }
printf 'completed_at=%s\niteration=50\nresume_verified_from=5\n' \
  "$(date -u +%FT%TZ)" > "${RUN}/SMOKE_COMPLETE"
echo "8-GPU 50-step native smoke and resume passed"
