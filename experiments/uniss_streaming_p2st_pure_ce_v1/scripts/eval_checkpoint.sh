#!/usr/bin/env bash
# Export one training checkpoint and re-run the cascade on it.
#
# This is the early-signal loop for the two criteria the run registered:
# the TTS terminator rate, 0.93 on the longest sample before training, and
# first_audible, which equalled the source end on three of four samples.
# Running it at the first saved checkpoint means a run that is not moving
# those numbers can be stopped in hours rather than at the end.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set the training RUN_ID}"
: "${ITER:?set the iteration to evaluate, e.g. 0000400}"

OWN_NAME=uniss_streaming_p2st_pure_ce_v1
SAVE_DIR=${REPO_ROOT}/checkpoints/${OWN_NAME}/${RUN_ID}
CHECKPOINT=${SAVE_DIR}/iter_${ITER}
HF_OUT=${REPO_ROOT}/checkpoints/exported_hf/${OWN_NAME}_${RUN_ID}_iter_${ITER}_hf
REPORT=${REPO_ROOT}/reports/${OWN_NAME}/${RUN_ID}/cascade_iter_${ITER}

[[ -f "${CHECKPOINT}/metadata.json" ]] || {
  echo "checkpoint not present: ${CHECKPOINT}" >&2; exit 3; }
mkdir -p "${REPORT}"

if [[ ! -f "${HF_OUT}/model.safetensors" ]]; then
  echo "step=export"
  # The same converter the established chain uses, unchanged.
  "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
    --hf-model "${V1_HF_MODEL}" \
    --megatron-path "${CHECKPOINT}" \
    --hf-output "${HF_OUT}" \
    --model-type gpt >/dev/null
fi

echo "step=cascade"
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" -m \
  experiments.uniss_streaming_p2st_pure_ce_v1.evaluation.cascade_mechanics \
  --gold "${PROCESSED_ROOT}/source_events/valid_gold_trajectories.jsonl" \
  --candidate-hf "${HF_OUT}" \
  --v1-checkpoint "${V1_CHECKPOINT}" \
  --whispervq-model "${WHISPERVQ_MODEL}" \
  --samples "${SAMPLES:-8}" --max-blocks "${MAX_BLOCKS:-20}" \
  --tts-text-scope "${TTS_TEXT_SCOPE:-delta}" \
  --output "${REPORT}/CASCADE_MECHANICS.json" \
  | tee "${REPORT}/CASCADE_MECHANICS.txt"

echo "report=${REPORT}/CASCADE_MECHANICS.json"
