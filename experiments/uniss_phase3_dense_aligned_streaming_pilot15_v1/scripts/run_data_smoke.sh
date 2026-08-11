#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"

SMOKE_ROOT="${DATA_ROOT}/smoke128_v2"
mkdir -p "${SMOKE_ROOT}"

"${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions \
  --input-manifest "${FORMAL_TRAIN}" \
  --output "${SMOKE_ROOT}/dense_train.jsonl" \
  --marker "${SMOKE_ROOT}/PART_COMPLETE.json" \
  --split train --num-parts 1 --part-index 0 --limit 128 --fail-fast \
  --low-watermark-ms "${LOW_WATERMARK_MS}" \
  --target-buffer-ms "${TARGET_BUFFER_MS}" \
  --semantic-history-tokens "${SEMANTIC_HISTORY_TOKENS}" \
  --progress-interval 32

"${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.audit \
  --manifest "${SMOKE_ROOT}/dense_train.jsonl" \
  --output "${SMOKE_ROOT}/data_audit.json"

"${PYTHON}" -m pytest \
  experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/tests -v
