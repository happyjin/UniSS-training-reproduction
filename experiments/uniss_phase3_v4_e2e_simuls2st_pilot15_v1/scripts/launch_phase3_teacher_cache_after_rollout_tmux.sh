#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"

FORMAL_RUN_ID=${FORMAL_RUN_ID:-phase3_teacher_formal_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_phase3_teacher_after_rollout}
WATCH_LOG=${LOG_ROOT}/phase3_teacher_cache/${FORMAL_RUN_ID}_watcher.log
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${WATCH_LOG}" ]]; then
  echo "refusing to overwrite Phase3 teacher-cache watcher log: ${WATCH_LOG}" >&2
  exit 2
fi
mkdir -p "$(dirname -- "${WATCH_LOG}")"

command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} V1_FORMAL_RUN_ID=${V1_FORMAL_RUN_ID} FORMAL_RUN_ID=${FORMAL_RUN_ID} ROLLOUT_SESSION=${ROLLOUT_SESSION:-uniss_e2e_v1_rollout_formal} WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30} NUM_GPUS=${NUM_GPUS:-8} PHASE3_PROCESSES_PER_GPU=${PHASE3_PROCESSES_PER_GPU:-1} V1_PROCESSES_PER_GPU=${V1_PROCESSES_PER_GPU:-1} PHASE3_SMOKE_LIMIT=${PHASE3_SMOKE_LIMIT:-256} V1_SMOKE_LIMIT=${V1_SMOKE_LIMIT:-64} ${SCRIPT_DIR}/wait_for_v1_then_run_phase3_teacher_cache.sh > ${WATCH_LOG} 2>&1"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "formal_run_id=${FORMAL_RUN_ID}"
echo "watch_log=${WATCH_LOG}"
