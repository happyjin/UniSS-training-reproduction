#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

STAGE_A_CHECKPOINT=${STAGE_A_CHECKPOINT:?STAGE_A_CHECKPOINT is required}
HF_MODEL=${HF_MODEL:?HF_MODEL is required}
VALID_PACKS=${VALID_PACKS:?VALID_PACKS is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
GPU=${GPU:-0}
MAX_SAMPLES_PER_TASK=${MAX_SAMPLES_PER_TASK:-2}
MAX_ACOUSTICS_PER_PACK=${MAX_ACOUSTICS_PER_PACK:-2}
OUTPUT_ROOT="${REPORT_ROOT}/stage_a_checkpoint_diagnosis/${RUN_ID}"

for required in "${HF_MODEL}/model.safetensors" "${VALID_PACKS}"; do
  [[ -e "${required}" ]] || { echo "missing Stage A v2 diagnosis input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite diagnosis: ${OUTPUT_ROOT}" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.evaluate_checkpoint \
  --checkpoint "${STAGE_A_CHECKPOINT}" \
  --hf-model "${HF_MODEL}" \
  --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
  --valid-packs "${VALID_PACKS}" \
  --max-samples-per-task "${MAX_SAMPLES_PER_TASK}" \
  --max-acoustics-per-pack "${MAX_ACOUSTICS_PER_PACK}" \
  --device cuda:0 \
  --output-json "${OUTPUT_ROOT}/diagnosis.json" \
  --output-md "${OUTPUT_ROOT}/diagnosis.md"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
