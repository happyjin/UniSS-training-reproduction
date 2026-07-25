#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
SESSION="simul_uniss_v3_full198_components_8gpu"
QWEN_SESSION="simul_uniss_v3_full198_qwen_8gpu"
QWEN_MARKER="${REPO_ROOT}/runs/simul_uniss_v3_full198/qwen_pipeline_8gpu/QWEN_PIPELINE_COMPLETE"
LOG="${REPO_ROOT}/logs/simul_uniss_v3_full198/component_pipeline_launcher.log"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
mkdir -p "$(dirname "${LOG}")"
wait_command="while [[ ! -f $(printf '%q' "${QWEN_MARKER}") ]]; do if ! tmux has-session -t $(printf '%q' "${QWEN_SESSION}") 2>/dev/null; then echo 'Qwen session ended without marker' >&2; exit 1; fi; sleep 30; done; $(printf '%q' "${EXPERIMENT_DIR}/orchestration/run_component_pipeline_8gpu.sh")"
command="cd $(printf '%q' "${REPO_ROOT}") && ${wait_command} 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; waiting for ${QWEN_MARKER}"
