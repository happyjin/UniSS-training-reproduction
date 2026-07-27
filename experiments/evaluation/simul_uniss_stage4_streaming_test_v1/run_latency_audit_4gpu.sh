#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PARENT_RUN_DIR" >&2
  exit 2
fi
PARENT_RUN_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

LATENCY_RECORDS="${LATENCY_RECORDS:-200}"
RUN_DIR="${PARENT_RUN_DIR}/latency_batch1"
mkdir -p "${RUN_DIR}/logs"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 >"${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

BATCH_RECORDS=1 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=4096 \
MAX_SEQ_LEN_TO_CAPTURE=4096 GPU_MEMORY_UTILIZATION=0.50 \
  "${SCRIPT_DIR}/run_generation_4gpu.sh" "${RUN_DIR}" "${LATENCY_RECORDS}"
DECODE_BATCH_SIZE=1 \
  "${SCRIPT_DIR}/run_decode_4gpu.sh" "${RUN_DIR}" "${LATENCY_RECORDS}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_DIR}/metrics"
"${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
  --input "${RUN_DIR}/generation_results.jsonl" \
  --output "${RUN_DIR}/metrics/text_bleu.json"
"${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
  --input "${RUN_DIR}/results.jsonl" \
  --output-dir "${RUN_DIR}/metrics"
"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" \
  --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_TEST}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/stage4_streaming_test_latency_batch1_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" \
  --gpu-ids "${TEST_GPUS}" \
  --split-label test-latency-batch1 \
  --expected-records "${LATENCY_RECORDS}"

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
