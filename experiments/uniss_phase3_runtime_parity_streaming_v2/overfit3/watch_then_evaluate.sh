#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit3/config.env"
export USER_ROOT REPO_ROOT ENV_ROOT PYTHON EXPERIMENT_NAME SAVE_DIR COVERAGE_EPOCHS
export EVAL_FAMILY=overfit3_v1 EXPORT_FAMILY=overfit3_v1
exec bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/watch_then_evaluate.sh"
