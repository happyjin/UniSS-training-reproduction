#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing Stage4 run: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/environment" "${RUN_DIR}/logs"

"${SCRIPT_DIR}/verify_dev_inputs.sh"
"${SCRIPT_DIR}/export_exact_stage4.sh"

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

"${SCRIPT_DIR}/run_generation_4gpu.sh" "${RUN_DIR}" 0
"${SCRIPT_DIR}/run_decode_4gpu.sh" "${RUN_DIR}" "${EXPECTED_DEV_RECORDS}"
"${SCRIPT_DIR}/run_common_metrics_4gpu.sh" "${RUN_DIR}" "${EXPECTED_DEV_RECORDS}"
"${SCRIPT_DIR}/run_latency_audit_4gpu.sh" "${RUN_DIR}"

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"
echo "REPORT=${RUN_DIR}/stage4_streaming_report.md"
