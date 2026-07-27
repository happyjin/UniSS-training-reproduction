#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

printf -v ITER_TAG 'iter_%07d' "$((10#${STAGE6_ITERATION}))"
SOURCE="${STAGE6_ROOT}/${ITER_TAG}"
TRACKER="${STAGE6_ROOT}/latest_checkpointed_iteration.txt"

[[ -x "${TRAIN_ENV}/bin/python" ]] || { echo "Missing train environment: ${TRAIN_ENV}" >&2; exit 1; }
[[ -d "${SOURCE}" && -f "${TRACKER}" ]] || { echo "Missing Stage6 checkpoint" >&2; exit 1; }
[[ "$(<"${TRACKER}")" == "${STAGE6_ITERATION}" ]] || { echo "Stage6 tracker mismatch" >&2; exit 1; }
shard_count="$(find "${SOURCE}" -maxdepth 1 -type f -name '__*_0.distcp' | wc -l)"
[[ "${shard_count}" -eq 8 ]] || { echo "Stage6 checkpoint has ${shard_count} shards, expected 8" >&2; exit 1; }

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
