#!/usr/bin/env bash
# Export one training checkpoint and re-run the cascade on it.
#
# This is the early-signal loop for the two criteria the run registered:
# the TTS terminator rate, 0.93 on the longest sample before training, and
# first_audible, which equalled the source end on three of four samples.
#
# MAX_BLOCKS defaults to 70 rather than 20 because at 20 the second criterion
# is unmeasurable: 20 blocks x BLOCK_MS 160 = 3200 ms of audio, and the
# iter_0000400 panel reported first_audible = 3200 ms on six of eight samples,
# i.e. exactly the window.  The observable was saturated by the harness, not by
# the model.  70 blocks is 11.2 s, which covers the longest panel source
# (10620 ms), so "did anything come out before the source ended" becomes a
# question about the model again.
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

# The converter and the cascade both load Transformer Engine, so they need the
# same library search order run_p2st_megatron.sh:118 establishes.  Without it
# the export step dies on libcudnn_graph.so.9, which is what happened to the
# queued iter_0000400 evaluation.
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR="${USER_ROOT}/tmp"
NVIDIA_LIBRARY_ROOT="$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia"
NVIDIA_LIBRARY_PATH=
if [[ -d "${NVIDIA_LIBRARY_ROOT}" ]]; then
  NVIDIA_LIBRARY_PATH=$(find "${NVIDIA_LIBRARY_ROOT}" \
    -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
fi
SYSTEM_CUDA_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${SYSTEM_CUDA_LIBRARY_PATH}:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}${NVIDIA_LIBRARY_PATH:+:${NVIDIA_LIBRARY_PATH}}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TMPDIR}"

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
  --samples "${SAMPLES:-8}" --max-blocks "${MAX_BLOCKS:-70}" \
  --tts-text-scope "${TTS_TEXT_SCOPE:-delta}" \
  --output "${REPORT}/CASCADE_MECHANICS.json" \
  | tee "${REPORT}/CASCADE_MECHANICS.txt"

echo "report=${REPORT}/CASCADE_MECHANICS.json"
