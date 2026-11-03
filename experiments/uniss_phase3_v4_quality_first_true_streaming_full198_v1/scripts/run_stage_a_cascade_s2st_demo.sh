#!/usr/bin/env bash
set -euo pipefail

# Stage A cascade S2ST demo.
#
# Custom audio (auto long-audio if longer than 15s):
#   AUDIO=/path/to.wav SRC_LANG=cmn TGT_LANG=eng \
#     bash scripts/run_stage_a_cascade_s2st_demo.sh
#
# Force long-audio / old short path:
#   LONG_AUDIO=1 ...   or   FORCE_SHORT=1 ...
#
# Optional: GPU, CHUNK_MS="160 320 640 1280" (any subset), RUN_ID, SAMPLE_NAME

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

ITERATION=${ITERATION:-4797}
FORMAL_RUN_ID=${FORMAL_RUN_ID:-stage_a_formal8_20260903T054603Z}
GPU=${GPU:-0}
CHUNK_MS=${CHUNK_MS:-"160 320 640 1280"}
read -r -a CHUNK_VALUES <<< "${CHUNK_MS}"

printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"
CHECKPOINT="${CHECKPOINT_ROOT}/stage_a_formal/${FORMAL_RUN_ID}/${ITER_TAG}"
HF_MODEL=${HF_MODEL:-"${REPO_ROOT}/checkpoints/exported_hf/${EXPERIMENT_NAME}_${FORMAL_RUN_ID}_${ITER_TAG}_hf"}
BICODEC=${BICODEC:-"${REPO_ROOT}/pretrained_models/UniSS/bicodec"}
COMPARE_MANIFEST="${SCRIPT_DIR}/cascade_s2st_compare_manifest.jsonl"

if [[ -n "${AUDIO:-}" ]]; then
  SRC_LANG=${SRC_LANG:?SRC_LANG is required with AUDIO}
  TGT_LANG=${TGT_LANG:?TGT_LANG is required with AUDIO}
  SAMPLE_NAME=${SAMPLE_NAME:-$(basename "${AUDIO%.*}")}
  RUN_ID=${RUN_ID:-iter${ITERATION}_custom_${SAMPLE_NAME}_$(date -u +%Y%m%dT%H%M%SZ)}
  OUTPUT="${EVAL_ROOT}/stage_a_cascade_s2st/${RUN_ID}"
  EXTRA=(
    --audio "${AUDIO}"
    --src-lang "${SRC_LANG}"
    --tgt-lang "${TGT_LANG}"
    --sample-name "${SAMPLE_NAME}"
  )
else
  RUN_ID=${RUN_ID:-iter${ITERATION}_vs15_demo_$(date -u +%Y%m%dT%H%M%SZ)}
  OUTPUT="${EVAL_ROOT}/stage_a_cascade_s2st/${RUN_ID}"
  EXTRA=(
    --manifest "${COMPARE_MANIFEST}"
    --compare
  )
fi

if [[ -n "${LONG_AUDIO:-}" ]]; then
  EXTRA+=(--long-audio)
fi
if [[ -n "${FORCE_SHORT:-}" ]]; then
  EXTRA+=(--force-short)
fi

for required in \
  "${CHECKPOINT}/.metadata" \
  "${HF_MODEL}/model.safetensors" \
  "${WHISPERVQ_CHECKPOINT}/model.safetensors" \
  "${BICODEC}/config.yaml" \
  "${STAGE_A_SOURCE_SNAPSHOT}"
do
  [[ -e "${required}" ]] || { echo "Missing cascade input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

mkdir -p "$(dirname "${OUTPUT}")"
echo "OUTPUT=${OUTPUT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "HF_MODEL=${HF_MODEL}"
echo "CHUNK_MS=${CHUNK_MS}"
echo "GPU=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/run_strict_causal_cascade_s2st.py" \
  --checkpoint "${CHECKPOINT}" \
  --hf-model "${HF_MODEL}" \
  --whispervq "${WHISPERVQ_CHECKPOINT}" \
  --bicodec "${BICODEC}" \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --output "${OUTPUT}" \
  --chunk-ms "${CHUNK_VALUES[@]}" \
  "${EXTRA[@]}"
