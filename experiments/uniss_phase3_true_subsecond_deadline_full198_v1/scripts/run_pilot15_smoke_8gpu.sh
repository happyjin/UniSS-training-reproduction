#!/usr/bin/env bash
set -euo pipefail

# Isolated 15-shard native-Megatron validation.  This intentionally uses a
# separately materialized 4096-token copy so it can coexist with the full198
# trajectory cache workers.  Formal training and its 18000-token inputs are
# never modified by this launcher.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

NAME="${PILOT15_SMOKE8_NAME:-uniss_true_subsecond_pilot15_native_50step_v2}"
DATA="${REPO_ROOT}/data/megatron/uniss_true_subsecond_pilot15_v1/short4096/train50_v1"
TRAJECTORY_PACKED="${PILOT15_TRAJECTORY_PACKED:-${DATA}/packed_trajectory.jsonl}"
TRAJECTORY_OFFSETS="${PILOT15_TRAJECTORY_OFFSETS:-${REPO_ROOT}/data/megatron/uniss_true_subsecond_pilot15_v1/short4096/train50_validated_v2/packed_trajectory.offsets.u64}"
REPLAY_PACKED="${PILOT15_REPLAY_PACKED:-${DATA}/packed_replay.jsonl}"
REPLAY_OFFSETS="${PILOT15_REPLAY_OFFSETS:-${DATA}/packed_replay.offsets.u64}"
SAVE="${REPO_ROOT}/checkpoints/${NAME}"
RUN="${REPO_ROOT}/runs/${NAME}"
LOG="${REPO_ROOT}/logs/${NAME}.log"
mkdir -p "${RUN}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for file in \
  "${TRAJECTORY_PACKED}" \
  "${TRAJECTORY_OFFSETS}" \
  "${REPLAY_PACKED}" \
  "${REPLAY_OFFSETS}"; do
  [[ -f "${file}" ]] || { echo "Missing pilot15 smoke input: ${file}" >&2; exit 1; }
done

common=(
  "RUN_NAME=${NAME}"
  "RUN_SAVE_DIR=${SAVE}"
  "RUN_TB_DIR=${RUN}/tensorboard"
  "RUN_LOG=${LOG}"
  "RUN_TRAJECTORY_PACKED=${TRAJECTORY_PACKED}"
  "RUN_TRAJECTORY_OFFSETS=${TRAJECTORY_OFFSETS}"
  "RUN_REPLAY_PACKED=${REPLAY_PACKED}"
  "RUN_REPLAY_OFFSETS=${REPLAY_OFFSETS}"
  RUN_TRAIN_ITERS=50 RUN_NPROC=8 RUN_MBS=1 RUN_GBS=128
  RUN_SEQ_LENGTH=4096 RUN_WARMUP_ITERS=5
  RUN_SAVE_INTERVAL=25 RUN_LOG_INTERVAL=1
  RUN_MASTER_PORT=29713 RUN_SMOKE=1 RUN_AUDIT_GRADIENTS=0 NUM_WORKERS=0
)

tracker=-1
[[ -s "${SAVE}/latest_checkpointed_iteration.txt" ]] && \
  tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
if [[ -f "${RUN}/SMOKE_COMPLETE" && "${tracker}" == "50" ]]; then
  echo "${NAME} already passed"
  exit 0
fi

# Phase A: non-strict Phase3 handoff, then save and deliberately exit at 5.
if (( tracker < 0 )); then
  [[ ! -e "${SAVE}" ]] || {
    echo "Refusing untracked pilot15 smoke directory: ${SAVE}" >&2
    exit 1
  }
  env "${common[@]}" RUN_LOAD="${PHASE3_NATIVE_CHECKPOINT}" \
    RUN_FINETUNE=1 RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 \
    RUN_STRICTNESS=log_all RUN_EXIT_INTERVAL=5 \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
  tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
  [[ "${tracker}" == "5" ]] || {
    echo "expected interruption checkpoint 5, got ${tracker}" >&2
    exit 1
  }
  printf 'phase3_handoff_complete_at=%s\ncheckpoint_iteration=5\n' \
    "$(date -u +%FT%TZ)" > "${RUN}/PHASE3_HANDOFF_COMPLETE"
fi

# Phase B: strict optimizer/RNG/scheduler/sampler restoration through step 50.
if (( tracker < 50 )); then
  env "${common[@]}" RUN_LOAD="${SAVE}" \
    RUN_FINETUNE=0 RUN_LOAD_OPTIM=1 RUN_LOAD_RNG=1 \
    RUN_STRICTNESS=raise_all \
    bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
fi

tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
[[ "${tracker}" == "50" ]] || {
  echo "8-GPU pilot15 smoke stopped at ${tracker}" >&2
  exit 1
}
if grep -Eq 'grad norm: (nan|inf|-inf)' "${LOG}"; then
  echo "non-finite gradient norm found in ${LOG}" >&2
  exit 1
fi
if grep -Eq 'number of (skipped|nan) iterations: +[1-9]' "${LOG}"; then
  echo "skipped or NaN iteration found in ${LOG}" >&2
  exit 1
fi
printf 'completed_at=%s\niteration=50\nresume_verified_from=5\nseq_length=4096\nsource=pilot15\n' \
  "$(date -u +%FT%TZ)" > "${RUN}/SMOKE_COMPLETE"
echo "8-GPU pilot15 50-step native smoke and strict resume passed"
