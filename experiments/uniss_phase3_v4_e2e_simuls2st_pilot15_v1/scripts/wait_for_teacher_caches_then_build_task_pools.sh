#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"
: "${TEACHER_FORMAL_RUN_ID:?set the immutable teacher-cache formal run ID}"
: "${TASK_POOL_RUN_ID:?set the immutable formal task-pool run ID}"

TEACHER_SESSION=${TEACHER_SESSION:-uniss_e2e_phase3_teacher_after_rollout}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30}
TASK_POOL_WORKERS=${TASK_POOL_WORKERS:-64}
if (( WAIT_INTERVAL_SECONDS < 1 )); then
  echo "WAIT_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi

audits=(
  "${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_FORMAL_RUN_ID}_phase3_train/AUDIT.json"
  "${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_FORMAL_RUN_ID}_phase3_valid/AUDIT.json"
  "${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_FORMAL_RUN_ID}_v1_train/AUDIT.json"
  "${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_FORMAL_RUN_ID}_v1_valid/AUDIT.json"
)

while true; do
  missing=0
  for path in "${audits[@]}"; do
    [[ -f "${path}" ]] || missing=$((missing + 1))
  done
  (( missing > 0 )) || break
  if ! tmux has-session -t "${TEACHER_SESSION}" 2>/dev/null; then
    echo "teacher-cache session ended with ${missing} audit(s) missing" >&2
    printf 'missing_or_pending=%s\n' "${audits[@]}" >&2
    exit 3
  fi
  printf '%s waiting for four teacher-cache audits (%d missing)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${missing}"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

for path in "${audits[@]}"; do
  jq -e '.status == "passed"' "${path}" >/dev/null || {
    echo "teacher-cache audit did not pass: ${path}" >&2
    exit 4
  }
done

printf '%s starting formal train/valid task-pool construction\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec env \
  DATA_RUN_ID="${DATA_RUN_ID}" \
  V1_FORMAL_RUN_ID="${V1_FORMAL_RUN_ID}" \
  TASK_POOL_RUN_ID="${TASK_POOL_RUN_ID}" \
  TASK_POOL_WORKERS="${TASK_POOL_WORKERS}" \
  "${SCRIPT_DIR}/run_formal_task_pools.sh"
