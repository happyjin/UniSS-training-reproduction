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
NPROC=${NPROC:-8}
MAX_ACOUSTICS_PER_PACK=${MAX_ACOUSTICS_PER_PACK:-2}
OUTPUT_ROOT="${REPORT_ROOT}/stage_a_checkpoint_diagnosis/${RUN_ID}"
PARTS="${OUTPUT_ROOT}/parts"

(( NPROC > 0 && NPROC <= 8 )) || { echo "NPROC must be in 1..8" >&2; exit 1; }
for required in "${HF_MODEL}/model.safetensors" "${VALID_PACKS}"; do
  [[ -e "${required}" ]] || { echo "missing Stage A v2 diagnosis input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite diagnosis: ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${PARTS}"

pids=()
for ((worker=0; worker<NPROC; worker++)); do
  printf -v tag '%02d' "${worker}"
  CUDA_VISIBLE_DEVICES="${worker}" PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.evaluate_checkpoint \
    --checkpoint "${STAGE_A_CHECKPOINT}" \
    --hf-model "${HF_MODEL}" \
    --whispervq-model "${WHISPERVQ_CHECKPOINT}" \
    --valid-packs "${VALID_PACKS}" \
    --max-samples-per-task 0 \
    --max-acoustics-per-pack "${MAX_ACOUSTICS_PER_PACK}" \
    --worker-index "${worker}" \
    --num-workers "${NPROC}" \
    --device cuda:0 \
    --output-json "${PARTS}/part_${tag}.json" \
    --output-md "${PARTS}/part_${tag}.md" \
    >"${PARTS}/part_${tag}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || { tail -n 80 "${PARTS}"/part_??.log >&2 || true; exit 1; }

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.merge_checkpoint_diagnosis \
  --parts "${PARTS}"/part_??.json \
  --output-json "${OUTPUT_ROOT}/diagnosis.json" \
  --output-md "${OUTPUT_ROOT}/diagnosis.md"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
