#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( "$1" != "dev" && "$1" != "test" ) ]]; then
  echo "Usage: $0 dev|test [RUN_ID]" >&2
  exit 2
fi
SPLIT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

RUN_ID="${2:-full_${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing Stage6 ${SPLIT} run: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/environment" "${RUN_DIR}/logs"

"${SCRIPT_DIR}/verify_split_inputs.sh" "${SPLIT}"
"${SCRIPT_DIR}/export_exact_stage6.sh"

if [[ "${SPLIT}" == "dev" ]]; then
  EXPECTED="${EXPECTED_DEV_RECORDS}"
else
  EXPECTED="${EXPECTED_TEST_RECORDS}"
fi
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_DIR}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short >"${RUN_DIR}/environment/git_status.txt"
cp "${SCRIPT_DIR}/experiment.env" "${RUN_DIR}/environment/experiment.env"
nvidia-smi -q >"${RUN_DIR}/environment/nvidia_smi_q.txt"

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 >"${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

"${SCRIPT_DIR}/run_generation_4gpu.sh" "${RUN_DIR}" "${SPLIT}" 0
"${SCRIPT_DIR}/run_decode_4gpu.sh" "${RUN_DIR}" "${SPLIT}" "${EXPECTED}"
"${SCRIPT_DIR}/run_common_metrics_4gpu.sh" "${RUN_DIR}" "${SPLIT}" "${EXPECTED}"
"${SCRIPT_DIR}/run_latency_audit_4gpu.sh" "${RUN_DIR}" "${SPLIT}"

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"
echo "REPORT=${RUN_DIR}/stage6_streaming_${SPLIT}_report.md"
