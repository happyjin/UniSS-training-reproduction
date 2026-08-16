#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
mkdir -p "${STAGE_A_DATA_ROOT}" "${TMPDIR}"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.freeze_inputs \
  --train-manifest "${STAGE_A_SOURCE_TRAIN}" \
  --valid-manifest "${STAGE_A_SOURCE_VALID}" \
  --ctc-map-dir "${STAGE_A_CTC_MAP_ROOT}" \
  --stage00-gate "${REPORT_ROOT}/stage00_baseline/GATE_PASSED.json" \
  --native-checkpoint "${PHASE3_NATIVE_CHECKPOINT}" \
  --output "${STAGE_A_SOURCE_SNAPSHOT}"

