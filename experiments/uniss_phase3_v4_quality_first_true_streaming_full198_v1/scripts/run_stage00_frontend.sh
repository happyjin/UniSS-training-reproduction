#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:?RUN_ID must be set by the Stage 00 launcher}
STAGE_REPORT_DIR="${REPORT_ROOT}/stage00_baseline/${RUN_ID}"
STAGE_LOG_DIR="${LOG_ROOT}/stage00_baseline/${RUN_ID}"
if [[ -e "${STAGE_REPORT_DIR}" || -e "${STAGE_LOG_DIR}" ]]; then
  echo "refusing to overwrite Stage 00 run ${RUN_ID}" >&2
  exit 2
fi
mkdir -p "${STAGE_REPORT_DIR}" "${STAGE_LOG_DIR}" "${TMPDIR}"

"${PYTHON_BIN}" -m pytest -q "${EXPERIMENT_DIR}/tests" \
  | tee "${STAGE_LOG_DIR}/cpu_tests.log"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.audit_phase3_artifacts \
  --native-checkpoint "${PHASE3_NATIVE_CHECKPOINT}" \
  --hf-checkpoint "${PHASE3_HF_CHECKPOINT}" \
  --whispervq-checkpoint "${WHISPERVQ_CHECKPOINT}" \
  --output-json "${STAGE_REPORT_DIR}/canonical_artifacts.json" \
  | tee "${STAGE_LOG_DIR}/canonical_artifacts.log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.audit_frontend_real_pcm \
  --manifest "${PILOT15_VALID_MANIFEST}" \
  --row-index 0 \
  --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
  --device cuda:0 \
  --output-json "${STAGE_REPORT_DIR}/frontend_real_pcm.json" \
  --passed-marker "${STAGE_REPORT_DIR}/FRONTEND_GATE_PASSED.json" \
  | tee "${STAGE_LOG_DIR}/frontend_real_pcm.log"

ln -s "${RUN_ID}" "${REPORT_ROOT}/stage00_baseline/latest.tmp"
mv -T "${REPORT_ROOT}/stage00_baseline/latest.tmp" \
  "${REPORT_ROOT}/stage00_baseline/latest"
