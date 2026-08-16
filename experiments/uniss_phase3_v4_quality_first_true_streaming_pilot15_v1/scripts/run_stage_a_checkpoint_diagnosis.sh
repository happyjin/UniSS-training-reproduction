#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

ITERATION=${ITERATION:?ITERATION is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
FORMAL_RUN_ID=${FORMAL_RUN_ID:?FORMAL_RUN_ID is required}
GPU=${GPU:-0}
MAX_SAMPLES_PER_TASK=${MAX_SAMPLES_PER_TASK:-2}

printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"
CHECKPOINT="${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/${FORMAL_RUN_ID}/${ITER_TAG}"
HF_MODEL="${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_${ITER_TAG}_hf"
OUTPUT_ROOT="${REPORT_ROOT}/stage_a_checkpoint_diagnosis/${RUN_ID}"

for required in "${CHECKPOINT}/.metadata" "${HF_MODEL}/model.safetensors" "${DATA_ROOT}/stage_a_causal_asr/valid_packs_18k_v1.jsonl"; do
  [[ -e "${required}" ]] || { echo "Missing Stage A diagnosis input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite diagnosis: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --hf-model "${HF_MODEL}" \
  --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
  --valid-packs "${DATA_ROOT}/stage_a_causal_asr/valid_packs_18k_v1.jsonl" \
  --chunk-ms 960 1280 \
  --max-samples-per-task "${MAX_SAMPLES_PER_TASK}" \
  --output-json "${OUTPUT_ROOT}/diagnosis.json" \
  --output-md "${OUTPUT_ROOT}/diagnosis.md"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
