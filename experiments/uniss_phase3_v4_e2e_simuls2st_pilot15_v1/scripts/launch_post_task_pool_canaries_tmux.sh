#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

: "${TASK_POOL_RUN_ID:?set the immutable formal task-pool run ID}"
: "${TEACHER_FORMAL_RUN_ID:?set the immutable teacher-cache run ID}"

CANARY_RUN_ID=${CANARY_RUN_ID:-post_task_pool_canary_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_post_task_pool_canaries}
TASK_POOL_SESSION=${TASK_POOL_SESSION:-uniss_e2e_task_pools_after_teacher}
WATCH_LOG=${LOG_ROOT}/post_task_pool_canaries/${CANARY_RUN_ID}_watcher.log
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${WATCH_LOG}" ]]; then
  echo "refusing to overwrite post-task-pool watcher log: ${WATCH_LOG}" >&2
  exit 2
fi
mkdir -p "$(dirname -- "${WATCH_LOG}")"

command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID} TEACHER_FORMAL_RUN_ID=${TEACHER_FORMAL_RUN_ID} CANARY_RUN_ID=${CANARY_RUN_ID} TASK_POOL_SESSION=${TASK_POOL_SESSION} WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30} CANARY_MBS=${CANARY_MBS:-2} CANARY_GBS=${CANARY_GBS:-128} CANARY_NUM_WORKERS=${CANARY_NUM_WORKERS:-8} CANARY_MASTER_PORT_BASE=${CANARY_MASTER_PORT_BASE:-29810} ${SCRIPT_DIR}/wait_for_task_pools_then_run_canaries.sh >> ${WATCH_LOG} 2>&1"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "canary_run_id=${CANARY_RUN_ID}"
echo "watch_log=${WATCH_LOG}"
