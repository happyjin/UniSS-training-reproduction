#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
RUN_ID=${RUN_ID:?RUN_ID must name the Stage A data audit}
OUTPUT="${STAGE_A_DATA_ROOT}/audit_${RUN_ID}"
if [[ ! -f "${STAGE_A_SOURCE_SNAPSHOT}" ]]; then
  echo "missing immutable Stage A snapshot: ${STAGE_A_SOURCE_SNAPSHOT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite Stage A data audit: ${OUTPUT}" >&2
  exit 3
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.audit_data \
  --train-manifest "${STAGE_A_SOURCE_TRAIN}" \
  --valid-manifest "${STAGE_A_SOURCE_VALID}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --ctc-map-dir "${STAGE_A_CTC_MAP_ROOT}" \
  --output-dir "${OUTPUT}" \
  --train-workers "${STAGE_A_AUDIT_TRAIN_WORKERS:-30}" \
  --valid-workers "${STAGE_A_AUDIT_VALID_WORKERS:-8}" \
  | tee "${LOG_ROOT}/stage_a/data_audit_${RUN_ID}.log"

echo "stage_a_data_audit=${OUTPUT}"

