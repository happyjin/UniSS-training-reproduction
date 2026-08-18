#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
: "${V1_ROLLOUT_RUN_ID:?set V1_ROLLOUT_RUN_ID to an audited rollout run}"

TEACHER_RUN_ID=${TEACHER_RUN_ID:-phase3_teacher_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_phase3_teacher_smoke}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} V1_ROLLOUT_RUN_ID=${V1_ROLLOUT_RUN_ID} TEACHER_RUN_ID=${TEACHER_RUN_ID} TEACHER_SPLIT=${TEACHER_SPLIT:-valid} TEACHER_START_INDEX=${TEACHER_START_INDEX:-0} TEACHER_LIMIT=${TEACHER_LIMIT:-256} NUM_GPUS=${NUM_GPUS:-8} PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-1} ${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "teacher_run_id=${TEACHER_RUN_ID}"
