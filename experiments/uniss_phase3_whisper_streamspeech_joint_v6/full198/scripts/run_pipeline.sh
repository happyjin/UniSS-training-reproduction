#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

validate_full198_data
mkdir -p "${FULL198_STATUS_ROOT}"
STATUS_FILE="${FULL198_STATUS_ROOT}/status.txt"
GPU_LOG="${FULL198_STATUS_ROOT}/gpu_power_utility.csv"
PIPELINE_LOG="${FULL198_STATUS_ROOT}/pipeline.log"

if [[ -e "${STATUS_FILE}" ]]; then
  echo "Refusing to reuse pipeline status: ${STATUS_FILE}" >&2
  exit 1
fi

bash "${SCRIPT_ROOT}/monitor_gpu.sh" "${GPU_LOG}" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo "stage_a_started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${STATUS_FILE}"
bash "${SCRIPT_ROOT}/run_stage_a.sh"
echo "stage_a_complete $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${STATUS_FILE}"

echo "stage_b_started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${STATUS_FILE}"
bash "${SCRIPT_ROOT}/run_stage_b.sh"
echo "complete $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${STATUS_FILE}"
