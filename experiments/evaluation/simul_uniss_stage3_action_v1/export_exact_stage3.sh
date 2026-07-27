#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

printf -v ITER_TAG 'iter_%07d' "$((10#${STAGE3_ITERATION}))"
MEGATRON_PATH="${STAGE3_CHECKPOINT_ROOT}/${ITER_TAG}"
TRACKER="${STAGE3_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

[[ -x "${ENV_ROOT}/bin/python" ]] || { echo "Missing environment: ${ENV_ROOT}" >&2; exit 1; }
[[ -d "${HF_REFERENCE}" ]] || { echo "Missing HF reference: ${HF_REFERENCE}" >&2; exit 1; }
[[ -d "${MEGATRON_PATH}" ]] || { echo "Missing Stage3 checkpoint: ${MEGATRON_PATH}" >&2; exit 1; }
[[ -f "${TRACKER}" ]] || { echo "Missing tracker: ${TRACKER}" >&2; exit 1; }
[[ "$(<"${TRACKER}")" == "${STAGE3_ITERATION}" ]] || {
  echo "Tracker does not point to requested Stage3 iteration" >&2
  exit 1
}

if [[ -d "${HF_EXPORT}" ]]; then
  "${ENV_ROOT}/bin/python" \
    "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_hf_export.py" \
    --model "${HF_EXPORT}" \
    --source-checkpoint "${MEGATRON_PATH}" \
    --expected-model-vocab-size 180480 \
    --expected-tokenizer-size 180407
  echo "Reusing verified HF export: ${HF_EXPORT}"
  exit 0
fi

PARTIAL="${HF_EXPORT}.partial.$$"
cleanup() {
  if [[ -d "${PARTIAL}" ]]; then
    rm -rf -- "${PARTIAL}"
  fi
}
trap cleanup EXIT

"${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_REFERENCE}" \
  --megatron-path "${MEGATRON_PATH}" \
  --hf-output "${PARTIAL}" \
  --model-type gpt \
  --no-progress

mv "${PARTIAL}" "${HF_EXPORT}"

"${ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_hf_export.py" \
  --model "${HF_EXPORT}" \
  --source-checkpoint "${MEGATRON_PATH}" \
  --expected-model-vocab-size 180480 \
  --expected-tokenizer-size 180407

echo "HF_EXPORT=${HF_EXPORT}"

