#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="${EXPERIMENT_NAME}_stage00"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

if tmux has-session -t uniss_gpu_load_60 2>/dev/null; then
  echo "stopping explicitly authorized synthetic GPU load: uniss_gpu_load_60"
  tmux kill-session -t uniss_gpu_load_60
  for _ in $(seq 1 30); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -q '[0-9]'; then
      break
    fi
    sleep 1
  done
fi

mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d' | sort -u
)
if (( ${#gpu_pids[@]} > 0 )); then
  echo "refusing to kill unknown GPU processes: ${gpu_pids[*]}" >&2
  exit 3
fi

mkdir -p "${LOG_ROOT}/stage00_baseline" "${TMPDIR}"
LAUNCH_LOG="${LOG_ROOT}/stage00_baseline/launch_${RUN_ID}.log"
COMMAND="cd '${REPO_ROOT}' && RUN_ID='${RUN_ID}' CUDA_VISIBLE_DEVICES=0 bash '${EXPERIMENT_DIR}/scripts/run_stage00_frontend.sh' 2>&1 | tee '${LAUNCH_LOG}'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${COMMAND}\""
echo "session=${SESSION}"
echo "run_id=${RUN_ID}"
echo "log=${LAUNCH_LOG}"

