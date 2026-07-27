#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/environment" "${RUN_DIR}/logs"

"${SCRIPT_DIR}/prepare_manifests.sh"
"${SCRIPT_DIR}/export_exact_stage3.sh"

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${RUN_DIR}/environment/git_status.txt"
"${ENV_ROOT}/bin/python" -m pip freeze > "${RUN_DIR}/environment/pip_freeze.txt"
nvidia-smi -q > "${RUN_DIR}/environment/nvidia_smi_q.txt"
cp "${SCRIPT_DIR}/experiment.env" "${RUN_DIR}/environment/experiment.env"

MONITOR_PID=""
DEV_PID=""
EVAL_PID=""
cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits \
  -l 2 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"

"${SCRIPT_DIR}/run_split_4gpu.sh" dev "${RUN_DIR}/dev" 0 \
  > "${RUN_DIR}/logs/dev.log" 2>&1 &
DEV_PID="$!"
"${SCRIPT_DIR}/run_split_4gpu.sh" eval "${RUN_DIR}/eval" 0 \
  > "${RUN_DIR}/logs/eval.log" 2>&1 &
EVAL_PID="$!"

DEV_STATUS=0
EVAL_STATUS=0
wait "${DEV_PID}" || DEV_STATUS="$?"
wait "${EVAL_PID}" || EVAL_STATUS="$?"
if [[ "${DEV_STATUS}" != "0" || "${EVAL_STATUS}" != "0" ]]; then
  echo "Stage3 evaluation failed: dev=${DEV_STATUS}, eval=${EVAL_STATUS}" >&2
  exit 1
fi

cleanup
MONITOR_PID=""

"${ENV_ROOT}/bin/python" -m evaluation.simultaneous_streaming.stage3_aggregate \
  --run-dir "${RUN_DIR}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/stage3_action_evaluation_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" \
  --expected-dev-samples "${EXPECTED_DEV_SAMPLES}" \
  --expected-eval-samples "${EXPECTED_EVAL_SAMPLES}"

echo "RUN_DIR=${RUN_DIR}"
echo "REPORT=${RUN_DIR}/stage3_action_evaluation_report.md"

