#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="${EXPERIMENT_NAME}_stage00_offline"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d' | sort -u
)
if (( ${#gpu_pids[@]} > 0 )); then
  echo "refusing to kill unknown GPU processes: ${gpu_pids[*]}" >&2
  exit 3
fi
LOG="${LOG_ROOT}/stage00_baseline/offline_launch_${RUN_ID}.log"
mkdir -p "$(dirname "${LOG}")"
COMMAND="cd '${REPO_ROOT}' && RUN_ID='${RUN_ID}' bash '${EXPERIMENT_DIR}/scripts/run_stage00_offline_baseline_8gpu.sh' 2>&1 | tee '${LOG}'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${COMMAND}\""
echo "session=${SESSION}"
echo "run_id=${RUN_ID}"
echo "log=${LOG}"

