#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

ITERATION=${ITERATION:?ITERATION is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
FORMAL_RUN_ID=${FORMAL_RUN_ID:?FORMAL_RUN_ID is required}
NPROC=${NPROC:-8}
CHUNK_MS=${CHUNK_MS:-"160 320 640 1280"}
read -r -a CHUNK_VALUES <<< "${CHUNK_MS}"
(( NPROC > 0 && NPROC <= 8 )) || { echo "NPROC must be in 1..8" >&2; exit 1; }

printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"
CHECKPOINT="${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/${FORMAL_RUN_ID}/${ITER_TAG}"
HF_MODEL="${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_${ITER_TAG}_hf"
OUTPUT_ROOT="${REPORT_ROOT}/stage_a_checkpoint_diagnosis/${RUN_ID}"
PARTS="${OUTPUT_ROOT}/parts"

for required in "${CHECKPOINT}/.metadata" "${HF_MODEL}/model.safetensors" "${STAGE_A_VALID_PACKS}"; do
  [[ -e "${required}" ]] || { echo "Missing Stage A diagnosis input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite diagnosis: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${PARTS}"

pids=()
for ((worker=0; worker<NPROC; worker++)); do
  printf -v tag '%02d' "${worker}"
  CUDA_VISIBLE_DEVICES="${worker}" "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint \
    --checkpoint "${CHECKPOINT}" \
    --hf-model "${HF_MODEL}" \
    --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
    --valid-packs "${STAGE_A_VALID_PACKS}" \
    --chunk-ms "${CHUNK_VALUES[@]}" \
    --max-samples-per-task 0 \
    --worker-index "${worker}" \
    --num-workers "${NPROC}" \
    --output-json "${PARTS}/part_${tag}.json" \
    --output-md "${PARTS}/part_${tag}.md" \
    >"${PARTS}/part_${tag}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || { echo "Stage A distributed diagnosis worker failed" >&2; exit 1; }

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.merge_checkpoint_diagnosis \
  --parts "${PARTS}"/part_??.json \
  --output-json "${OUTPUT_ROOT}/diagnosis.json" \
  --output-md "${OUTPUT_ROOT}/diagnosis.md"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
