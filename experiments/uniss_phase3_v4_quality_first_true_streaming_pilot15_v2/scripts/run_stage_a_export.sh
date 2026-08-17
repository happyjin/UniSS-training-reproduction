#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

STAGE_A_CHECKPOINT=${STAGE_A_CHECKPOINT:?STAGE_A_CHECKPOINT is required}
HF_OUTPUT=${HF_OUTPUT:?HF_OUTPUT is required}
HF_ARCHITECTURE_REFERENCE=${HF_ARCHITECTURE_REFERENCE:-${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab_hf}

[[ -e "${STAGE_A_CHECKPOINT}" ]] || { echo "missing Stage A checkpoint: ${STAGE_A_CHECKPOINT}" >&2; exit 1; }
[[ -f "${HF_ARCHITECTURE_REFERENCE}/config.json" ]] || { echo "missing HF architecture reference" >&2; exit 1; }
[[ ! -e "${HF_OUTPUT}" ]] || { echo "refusing to overwrite HF export: ${HF_OUTPUT}" >&2; exit 2; }

ENV_ROOT="$(dirname "$(dirname "${PYTHON_BIN}")")" \
  "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_ARCHITECTURE_REFERENCE}" \
  --megatron-path "${STAGE_A_CHECKPOINT}" \
  --hf-output "${HF_OUTPUT}" \
  --model-type gpt \
  --no-progress

for required in model.safetensors config.json tokenizer.json; do
  [[ -s "${HF_OUTPUT}/${required}" ]] || { echo "incomplete HF export: ${required}" >&2; exit 3; }
done
echo "HF_OUTPUT=${HF_OUTPUT}"
