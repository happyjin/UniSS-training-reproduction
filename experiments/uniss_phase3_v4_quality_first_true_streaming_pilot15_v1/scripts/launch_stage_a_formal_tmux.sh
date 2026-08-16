#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

SESSION="${EXPERIMENT_NAME}_stage_a_formal"
RUN_ID=${RUN_ID:-stage_a_formal8_$(date -u +%Y%m%dT%H%M%SZ)}
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

COMMAND="cd '${REPO_ROOT}' && RUN_ID='${RUN_ID}' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash '${SCRIPT_DIR}/run_stage_a_formal_8gpu.sh'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${COMMAND}\""
echo "session=${SESSION}"
echo "run_id=${RUN_ID}"
echo "tensorboard=${RUN_ROOT}/stage_a_formal/${RUN_ID}/tensorboard"
echo "log=${LOG_ROOT}/stage_a_formal/${RUN_ID}/train.log"

