#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:?RUN_ID must name the existing Stage 00 report run}
HF_REFERENCE="${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab_hf"
REEXPORT="${CHECKPOINT_ROOT}/stage00_native_reexport_${RUN_ID}"
REPORT="${REPORT_ROOT}/stage00_baseline/${RUN_ID}/native_hf_reexport_parity.json"
if [[ -e "${REEXPORT}" || -e "${REPORT}" ]]; then
  echo "refusing to overwrite native re-export audit" >&2
  exit 2
fi
mkdir -p "${CHECKPOINT_ROOT}" "${TMPDIR}"

"${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_REFERENCE}" \
  --megatron-path "${PHASE3_NATIVE_CHECKPOINT}" \
  --hf-output "${REEXPORT}" \
  --model-type gpt \
  --strict \
  --no-progress

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.audit_native_hf_reexport \
  --canonical-hf "${PHASE3_HF_CHECKPOINT}" \
  --reexported-hf "${REEXPORT}" \
  --validation-manifest "${DATA_ROOT}/stage00_fixed_validation_v1/pilot15_text_256.jsonl" \
  --output-json "${REPORT}" \
  --device cuda:0

