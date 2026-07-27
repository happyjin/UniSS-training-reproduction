#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

printf -v ITER_TAG 'iter_%07d' "$((10#${STAGE4_ITERATION}))"
SOURCE="${STAGE4_ROOT}/${ITER_TAG}"
TRACKER="${STAGE4_ROOT}/latest_checkpointed_iteration.txt"

[[ -x "${TRAIN_ENV}/bin/python" ]] || { echo "Missing train environment: ${TRAIN_ENV}" >&2; exit 1; }
[[ -d "${SOURCE}" && -f "${TRACKER}" ]] || { echo "Missing Stage4 checkpoint" >&2; exit 1; }
[[ "$(<"${TRACKER}")" == "${STAGE4_ITERATION}" ]] || { echo "Stage4 tracker mismatch" >&2; exit 1; }

if [[ ! -d "${HF_EXPORT}" ]]; then
  PARTIAL="${HF_EXPORT}.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
    --hf-model "${HF_REFERENCE}" \
    --megatron-path "${SOURCE}" \
    --hf-output "${PARTIAL}" \
    --model-type gpt \
    --no-progress
  mv "${PARTIAL}" "${HF_EXPORT}"
fi

"${TRAIN_ENV}/bin/python" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_hf_export.py" \
  --model "${HF_EXPORT}" \
  --source-checkpoint "${SOURCE}" \
  --expected-model-vocab-size 180480 \
  --expected-tokenizer-size 180407

echo "HF_EXPORT=${HF_EXPORT}"
