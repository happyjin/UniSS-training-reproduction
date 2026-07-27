#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_DIR EXPECTED_RECORDS" >&2
  exit 2
fi
RUN_DIR="$1"
EXPECTED="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_DIR}/metrics"
if [[ ! -f "${RUN_DIR}/metrics/text_bleu.json" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
    --input "${RUN_DIR}/generation_results.jsonl" \
    --output "${RUN_DIR}/metrics/text_bleu.json"
fi
if [[ ! -f "${RUN_DIR}/metrics/slc.json" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
    --input "${RUN_DIR}/results.jsonl" \
    --output-dir "${RUN_DIR}/metrics"
fi

EVAL_GPU_LIST="${TEST_GPUS}" METRIC_NUM_GPUS=4 \
ENV_ROOT="${EVAL_ENV}" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh" \
  "${RUN_DIR}"

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" \
  --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_TEST}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/stage4_streaming_test_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" \
  --gpu-ids "${TEST_GPUS}" \
  --split-label test \
  --expected-records "${EXPECTED}"
