#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit3/config.env"
export USER_ROOT REPO_ROOT ENV_ROOT PYTHON SAVE_DIR
ITERATION="${ITERATION:-${COVERAGE_EPOCHS}}" TAG="${TAG:-natural_continuation_v1}" \
EVAL_FAMILY=overfit3_v1 EXPORT_FAMILY=overfit3_v1 \
exec bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/evaluate_checkpoint.sh"
