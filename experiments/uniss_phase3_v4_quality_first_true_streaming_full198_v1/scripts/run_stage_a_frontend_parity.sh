#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:?RUN_ID must be set}
REPORT_DIR="${REPORT_ROOT}/stage_a_causal_whisper_asr/frontend_parity/${RUN_ID}"
LOG_DIR="${LOG_ROOT}/stage_a_causal_whisper_asr/frontend_parity/${RUN_ID}"
if [[ -e "${REPORT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "refusing to overwrite Stage A frontend parity ${RUN_ID}" >&2
  exit 2
fi
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${TMPDIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.audit_trainable_frontend \
  --manifest "${PILOT15_VALID_MANIFEST}" \
  --row-index 0 \
  --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
  --device cuda:0 \
  --output-json "${REPORT_DIR}/trainable_frontend_parity.json" \
  --passed-marker "${REPORT_DIR}/FRONTEND_TRAINING_GATE_PASSED.json" \
  2>&1 | tee "${LOG_DIR}/trainable_frontend_parity.log"

echo "report=${REPORT_DIR}/trainable_frontend_parity.json"
echo "gate=${REPORT_DIR}/FRONTEND_TRAINING_GATE_PASSED.json"
