#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit4/config.env"
export USER_ROOT REPO_ROOT ENV_ROOT PYTHON SAVE_DIR COVERAGE_EPOCHS
export FUSE_TICKS=1 STATIC_CACHE=1 MAXIMUM_CACHE_TOKENS=32768
ITERATION="${ITERATION:-${COVERAGE_EPOCHS}}" TAG="${TAG:-content_consolidation_v1}" \
EVAL_FAMILY=overfit4_v1 EXPORT_FAMILY=overfit4_v1 \
exec bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/evaluate_checkpoint.sh"
