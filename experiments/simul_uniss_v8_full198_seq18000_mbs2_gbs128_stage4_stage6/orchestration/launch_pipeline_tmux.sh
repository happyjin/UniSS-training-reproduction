#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="simul_uniss_v8_seq18000_stage4_stage6_8gpu"
LOG="${LOG_DIR}/pipeline_launcher.log"
[[ -f "${FULL_DATA_READY_MARKER}" ]] || { echo "18k interleaved data is not ready" >&2; exit 1; }
[[ -f "${V7_STAGE3_ROOT}/latest_checkpointed_iteration.txt" ]] || { echo "v7 Stage3 checkpoint is missing" >&2; exit 1; }
[[ "$(tr -d '[:space:]' < "${V7_STAGE3_ROOT}/latest_checkpointed_iteration.txt")" == "${V7_STAGE3_REQUIRED_ITERATION}" ]] || {
  echo "v7 Stage3 is not complete" >&2; exit 1;
}
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/orchestration/run_stage4_stage6_8gpu.sh") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
