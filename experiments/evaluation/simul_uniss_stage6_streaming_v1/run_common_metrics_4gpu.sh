#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ( "$2" != "dev" && "$2" != "test" ) ]]; then
  echo "Usage: $0 RUN_DIR dev|test EXPECTED_RECORDS" >&2
  exit 2
fi
RUN_DIR="$1"
SPLIT="$2"
EXPECTED="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

if [[ "${SPLIT}" == "dev" ]]; then
  GPU_LIST="${DEV_GPUS}"
  OFFLINE_ROOT="${OFFLINE_PHASE3_DEV}"
else
  GPU_LIST="${TEST_GPUS}"
  OFFLINE_ROOT="${OFFLINE_PHASE3_TEST}"
fi
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

EVAL_GPU_LIST="${GPU_LIST}" METRIC_NUM_GPUS=4 ENV_ROOT="${EVAL_ENV}" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh" \
  "${RUN_DIR}"

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" \
  --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_ROOT}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/stage6_streaming_${SPLIT}_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" \
  --gpu-ids "${GPU_LIST}" \
  --split-label "${SPLIT}" \
  --stage-label Stage6 \
  --stage-iteration "${STAGE6_ITERATION}" \
  --stage-description "joint low-LR interleaved refinement" \
  --streaming-mode "${STREAMING_MODE}" \
  --expected-records "${EXPECTED}"
