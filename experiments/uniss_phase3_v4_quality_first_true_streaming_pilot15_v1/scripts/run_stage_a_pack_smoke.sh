#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT="${STAGE_A_DATA_ROOT}/pack_smoke_${RUN_ID}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite Stage A pack smoke: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${TMPDIR}" "${LOG_ROOT}/stage_a"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.build_training_packs \
  --manifest "${STAGE_A_SOURCE_TRAIN}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --output "${OUTPUT_ROOT}/train_packs.jsonl" \
  --workers 8 \
  --seq-length 4096 \
  --limit 256 \
  | tee "${LOG_ROOT}/stage_a/pack_smoke_${RUN_ID}.train.log"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.build_training_packs \
  --manifest "${STAGE_A_SOURCE_VALID}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --output "${OUTPUT_ROOT}/valid_packs.jsonl" \
  --workers 4 \
  --seq-length 4096 \
  --limit 64 \
  | tee "${LOG_ROOT}/stage_a/pack_smoke_${RUN_ID}.valid.log"

echo "stage_a_pack_smoke=${OUTPUT_ROOT}"
