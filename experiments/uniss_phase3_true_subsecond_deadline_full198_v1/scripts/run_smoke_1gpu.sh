#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

NAME="${SMOKE1_NAME:-uniss_true_subsecond_native_full198_smoke1_v12}"
SMOKE_ROOT="${SMOKE_ROOT:-${DATA_ROOT}/smoke/trajectory_assembly_v2}"
SMOKE_TRAJECTORY_PACKED="${SMOKE_TRAJECTORY_PACKED:-${SMOKE_ROOT}/packed_trajectory.jsonl}"
SMOKE_TRAJECTORY_OFFSETS="${SMOKE_TRAJECTORY_OFFSETS:-${SMOKE_ROOT}/packed_trajectory.offsets.u64}"
SMOKE_REPLAY_PACKED="${SMOKE_REPLAY_PACKED:-${PHASE3_REPLAY_PACKED}}"
SMOKE_REPLAY_OFFSETS="${SMOKE_REPLAY_OFFSETS:-${REPO_ROOT}/data/processed/phase3_whisper_streamspeech_joint_v1/phase3_replay_index/smoke_first8.u64}"
SAVE="${REPO_ROOT}/checkpoints/${NAME}"
RUN="${REPO_ROOT}/runs/${NAME}"
LOG="${REPO_ROOT}/logs/${NAME}.log"
if [[ -e "${SAVE}" || -e "${RUN}" ]]; then
  [[ -f "${RUN}/SMOKE_COMPLETE" ]] && { echo "${NAME} already passed"; exit 0; }
  echo "Refusing to overwrite incomplete smoke output: ${SAVE} or ${RUN}" >&2
  exit 1
fi
mkdir -p "${RUN}"
export CUDA_VISIBLE_DEVICES=0
RUN_NAME="${NAME}" \
RUN_SAVE_DIR="${SAVE}" \
RUN_TB_DIR="${RUN}/tensorboard" \
RUN_LOG="${LOG}" \
RUN_TRAJECTORY_PACKED="${SMOKE_TRAJECTORY_PACKED}" \
RUN_TRAJECTORY_OFFSETS="${SMOKE_TRAJECTORY_OFFSETS}" \
RUN_REPLAY_PACKED="${SMOKE_REPLAY_PACKED}" \
RUN_REPLAY_OFFSETS="${SMOKE_REPLAY_OFFSETS}" \
RUN_TRAIN_ITERS=1 RUN_NPROC=1 RUN_MBS=1 RUN_GBS=9 \
RUN_SEQ_LENGTH="${SMOKE_SEQ_LENGTH:-18000}" \
RUN_WARMUP_ITERS=0 RUN_SAVE_INTERVAL=1 RUN_LOG_INTERVAL=1 \
RUN_MASTER_PORT=29711 RUN_LOAD="${PHASE3_NATIVE_CHECKPOINT}" \
RUN_FINETUNE=1 RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 \
RUN_STRICTNESS=log_all RUN_SMOKE=1 RUN_AUDIT_GRADIENTS=1 NUM_WORKERS=0 \
bash "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh" "$@"
tracker="$(tr -d '[:space:]' < "${SAVE}/latest_checkpointed_iteration.txt")"
[[ "${tracker}" == "1" ]] || { echo "1-GPU smoke stopped at ${tracker}" >&2; exit 1; }
printf 'completed_at=%s\niteration=1\n' "$(date -u +%FT%TZ)" > "${RUN}/SMOKE_COMPLETE"
echo "1-GPU native smoke passed"
