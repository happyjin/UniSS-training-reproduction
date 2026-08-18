#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"
: "${TEACHER_FORMAL_RUN_ID:?set the immutable teacher-cache formal run ID}"

TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID:-task_pool_formal_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_task_pools_after_teacher}
TEACHER_SESSION=${TEACHER_SESSION:-uniss_e2e_phase3_teacher_after_rollout}
WATCH_LOG=${LOG_ROOT}/task_pools/${TASK_POOL_RUN_ID}_watcher.log
RESUME_WATCHER=${RESUME_WATCHER:-0}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${WATCH_LOG}" && "${RESUME_WATCHER}" != "1" ]]; then
  echo "refusing to overwrite task-pool watcher log: ${WATCH_LOG}" >&2
  exit 2
fi
mkdir -p "$(dirname -- "${WATCH_LOG}")"

redirect='>'
if [[ "${RESUME_WATCHER}" == "1" ]]; then
  redirect='>>'
  printf '%s restarting watcher with current gate logic\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${WATCH_LOG}"
fi
command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} V1_FORMAL_RUN_ID=${V1_FORMAL_RUN_ID} TEACHER_FORMAL_RUN_ID=${TEACHER_FORMAL_RUN_ID} TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID} TEACHER_SESSION=${TEACHER_SESSION} WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30} TASK_POOL_WORKERS=${TASK_POOL_WORKERS:-64} ${SCRIPT_DIR}/wait_for_teacher_caches_then_build_task_pools.sh ${redirect} ${WATCH_LOG} 2>&1"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "task_pool_run_id=${TASK_POOL_RUN_ID}"
echo "watch_log=${WATCH_LOG}"
