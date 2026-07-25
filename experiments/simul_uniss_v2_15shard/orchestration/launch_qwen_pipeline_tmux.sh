#!/usr/bin/env bash
set -euo pipefail

RECOVER_COMPLETED=0
if [[ "${1:-}" == "--recover-completed" ]]; then RECOVER_COMPLETED=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
SESSION="simul_uniss_v2_qwen_8gpu"
LOG="${REPO_ROOT}/logs/simul_uniss_v2_15shard/qwen_pipeline_launcher.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi
mkdir -p "$(dirname "${LOG}")"
pipeline=("${EXPERIMENT_DIR}/orchestration/run_qwen_pipeline_8gpu.sh")
[[ "${RECOVER_COMPLETED}" == "0" ]] || pipeline+=(--recover-completed)
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q ' "${pipeline[@]}") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; launcher log: ${LOG}"
