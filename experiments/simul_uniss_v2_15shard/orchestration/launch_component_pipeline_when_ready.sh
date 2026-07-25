#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"

SESSION="simul_uniss_v2_components_8gpu"
QWEN_SESSION="simul_uniss_v2_qwen_8gpu"
QWEN_MARKER="${RUN_DIR}/qwen_pipeline_8gpu/QWEN_PIPELINE_COMPLETE"
LOG="${LOG_DIR}/component_pipeline_launcher.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi
mkdir -p "${LOG_DIR}"
wait_command="while [[ ! -f $(printf '%q' "${QWEN_MARKER}") ]]; do if ! tmux has-session -t $(printf '%q' "${QWEN_SESSION}") 2>/dev/null; then echo 'Qwen session ended without completion marker' >&2; exit 1; fi; echo \"[\$(date -u +%FT%TZ)] waiting for Qwen pipeline\"; sleep 30; done; $(printf '%q' "${EXPERIMENT_DIR}/orchestration/run_component_pipeline_8gpu.sh")"
command="cd $(printf '%q' "${REPO_ROOT}") && ${wait_command} 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; it will wait for ${QWEN_MARKER}"

