#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config.env"

[[ -f "${DEV_SOURCE}" ]] || { echo "Missing canonical UniST dev: ${DEV_SOURCE}" >&2; exit 1; }
mkdir -p "${DEV_INDEX_ROOT}" "${DEV_PLAN_ROOT}"

"${PYTHON}" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_dev_direction_index \
  --input "${DEV_SOURCE}" \
  --output-root "${DEV_INDEX_ROOT}" \
  --partitions "${DEV_SHARD_COUNT}"

"${PYTHON}" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_schedule \
  --index-json "${DEV_INDEX_ROOT}/index.json" \
  --output-root "${DEV_PLAN_ROOT}" \
  --workers "${DEV_SHARD_COUNT}" \
  --shard-count "${DEV_SHARD_COUNT}" \
  --index-template 'part-{shard:03d}.{lang}.npy'

echo "canonical dev index and deterministic bidirectional trajectory plan are ready"
