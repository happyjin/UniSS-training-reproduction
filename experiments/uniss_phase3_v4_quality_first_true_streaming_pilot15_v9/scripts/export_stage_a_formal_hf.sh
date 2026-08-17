#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

FORMAL_RUN_ID=${FORMAL_RUN_ID:?FORMAL_RUN_ID is required}
ITERATION=${ITERATION:-381}
printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"

MEGATRON_PATH="${CHECKPOINT_ROOT}/stage_a_formal/${FORMAL_RUN_ID}/${ITER_TAG}"
HF_REFERENCE="${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab_hf"
HF_OUTPUT=${HF_OUTPUT:-"${REPO_ROOT}/checkpoints/exported_hf/${EXPERIMENT_NAME}_${FORMAL_RUN_ID}_${ITER_TAG}_hf"}

[[ -f "${MEGATRON_PATH}/.metadata" ]] || {
  echo "Missing V9 formal checkpoint: ${MEGATRON_PATH}" >&2
  exit 1
}
[[ -f "${HF_REFERENCE}/config.json" ]] || {
  echo "Missing Qwen HF reference: ${HF_REFERENCE}" >&2
  exit 1
}
[[ ! -e "${HF_OUTPUT}" ]] || {
  echo "Refusing to overwrite HF export: ${HF_OUTPUT}" >&2
  exit 1
}

exec bash "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_REFERENCE}" \
  --megatron-path "${MEGATRON_PATH}" \
  --hf-output "${HF_OUTPUT}" \
  --model-type gpt \
  --no-progress
