#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

FORMAL_RUN_ID=${FORMAL_RUN_ID:?FORMAL_RUN_ID is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
ITERATION=${ITERATION:-381}
NPROC=${NPROC:-8}
CHUNK_MS=${CHUNK_MS:-"160 320 640 1280"}
MAX_SAMPLES_PER_TASK=${MAX_SAMPLES_PER_TASK:-0}
read -r -a CHUNK_VALUES <<< "${CHUNK_MS}"
printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"

CHECKPOINT="${CHECKPOINT_ROOT}/stage_a_formal/${FORMAL_RUN_ID}/${ITER_TAG}"
HF_MODEL=${HF_MODEL:-"${REPO_ROOT}/checkpoints/exported_hf/${EXPERIMENT_NAME}_${FORMAL_RUN_ID}_${ITER_TAG}_hf"}
WHISPERVQ_MODEL="${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer"
VALID_PACKS="${REPO_ROOT}/data/megatron/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/valid_packs_18k_v1.jsonl"
OUTPUT_ROOT="${REPORT_ROOT}/stage_a_free_running_eval/${RUN_ID}"
PARTS="${OUTPUT_ROOT}/parts"

(( NPROC > 0 && NPROC <= 8 )) || {
  echo "NPROC must be in 1..8" >&2
  exit 1
}
for required in "${CHECKPOINT}/.metadata" "${HF_MODEL}/model.safetensors" "${VALID_PACKS}"; do
  [[ -e "${required}" ]] || {
    echo "Missing V9 evaluation input: ${required}" >&2
    exit 1
  }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "Refusing to overwrite V9 evaluation: ${OUTPUT_ROOT}" >&2
  exit 1
}
mkdir -p "${PARTS}"

pids=()
for ((worker=0; worker<NPROC; worker++)); do
  printf -v tag '%02d' "${worker}"
  CUDA_VISIBLE_DEVICES="${worker}" "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.evaluate_checkpoint \
    --checkpoint "${CHECKPOINT}" \
    --hf-model "${HF_MODEL}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --valid-packs "${VALID_PACKS}" \
    --chunk-ms "${CHUNK_VALUES[@]}" \
    --max-samples-per-task "${MAX_SAMPLES_PER_TASK}" \
    --max-acoustics-per-pack 2 \
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
(( failed == 0 )) || {
  echo "V9 free-running evaluation worker failed" >&2
  exit 1
}

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.merge_checkpoint_diagnosis \
  --parts "${PARTS}"/part_??.json \
  --output-json "${OUTPUT_ROOT}/diagnosis.json" \
  --output-md "${OUTPUT_ROOT}/diagnosis.md"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
