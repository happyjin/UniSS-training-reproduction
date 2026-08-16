#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="${EXPERIMENT_NAME}_stage_a_data"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ ! -f "${REPORT_ROOT}/stage00_baseline/GATE_PASSED.json" ]]; then
  echo "Stage 00 gate is missing" >&2
  exit 3
fi
if [[ ! -f "${STAGE_A_SOURCE_SNAPSHOT}" ]]; then
  echo "Stage A source snapshot is missing" >&2
  exit 4
fi
LOG="${LOG_ROOT}/stage_a/data_audit_launch_${RUN_ID}.log"
mkdir -p "$(dirname "${LOG}")"
COMMAND="cd '${REPO_ROOT}' && RUN_ID='${RUN_ID}' bash '${EXPERIMENT_DIR}/scripts/run_stage_a_data_audit.sh' 2>&1 | tee '${LOG}'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${COMMAND}\""
echo "session=${SESSION}"
echo "run_id=${RUN_ID}"
echo "log=${LOG}"
