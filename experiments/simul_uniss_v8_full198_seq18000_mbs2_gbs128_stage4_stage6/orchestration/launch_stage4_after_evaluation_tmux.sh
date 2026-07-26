#!/usr/bin/env bash
set -euo pipefail

EVALUATION_ROOT="${1:-}"
[[ -n "${EVALUATION_ROOT}" ]] || { echo "Usage: $0 EVALUATION_CONTROL_ROOT" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="simul_uniss_v8_seq18000_stage4_after_evaluation"
LOG="${LOG_DIR}/stage4_after_evaluation_launcher.log"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/orchestration/run_stage4_after_evaluation.sh") --evaluation-root $(printf '%q' "${EVALUATION_ROOT}") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
