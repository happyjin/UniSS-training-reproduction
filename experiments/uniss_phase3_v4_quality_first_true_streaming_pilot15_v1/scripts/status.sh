#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

echo "experiment=${EXPERIMENT_NAME}"
echo "report_root=${REPORT_ROOT}"
echo "log_root=${LOG_ROOT}"
if tmux has-session -t "${EXPERIMENT_NAME}_stage00" 2>/dev/null; then
  echo "stage00_tmux=running"
else
  echo "stage00_tmux=not_running"
fi
if tmux has-session -t "${EXPERIMENT_NAME}_stage00_offline" 2>/dev/null; then
  echo "stage00_offline_tmux=running"
else
  echo "stage00_offline_tmux=not_running"
fi
if [[ -f "${REPORT_ROOT}/stage00_baseline/latest/FRONTEND_GATE_PASSED.json" ]]; then
  echo "frontend_gate=passed"
else
  echo "frontend_gate=not_passed"
fi
if [[ -f "${REPORT_ROOT}/stage00_baseline/GATE_PASSED.json" ]]; then
  echo "stage00_gate=passed"
else
  echo "stage00_gate=not_passed"
fi
if [[ -f "${REPORT_ROOT}/stage00_baseline/STAGE00_RESULT_REPORT.md" ]]; then
  echo "stage00_report=${REPORT_ROOT}/stage00_baseline/STAGE00_RESULT_REPORT.md"
fi
if [[ -f "${LOG_ROOT}/stage00_baseline/frontend_real_pcm.log" ]]; then
  tail -n 20 "${LOG_ROOT}/stage00_baseline/frontend_real_pcm.log"
fi
