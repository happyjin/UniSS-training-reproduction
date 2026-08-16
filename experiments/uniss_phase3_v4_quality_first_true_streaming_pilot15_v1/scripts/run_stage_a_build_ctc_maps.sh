#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
mkdir -p "${STAGE_A_DATA_ROOT}" "${TMPDIR}" "${LOG_ROOT}/stage_a"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.build_ctc_maps \
  --train-manifest "${STAGE_A_SOURCE_TRAIN}" \
  --valid-manifest "${STAGE_A_SOURCE_VALID}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --output-dir "${STAGE_A_CTC_MAP_ROOT}" \
  --reference-map-dir "${STAGE_A_REFERENCE_CTC_MAP_ROOT}" \
  --target-kind utf8_byte \
  --train-workers "${STAGE_A_CTC_TRAIN_WORKERS:-30}" \
  --valid-workers "${STAGE_A_CTC_VALID_WORKERS:-8}" \
  | tee "${LOG_ROOT}/stage_a/ctc_map_build_v4.log"
