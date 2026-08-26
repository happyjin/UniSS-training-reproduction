#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
SESSION=${GPU_HOLDER_SESSION:-uniss_gpu_load_60}
REPORT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/REPORT.zh-CN.md
LOG=${REPO_ROOT}/logs/uniss_phasea_stateful_longepisode_rl_v1/gpu_holder_after_completion.log
HOLDER=${REPO_ROOT}/scripts/gpu_load/target_gpu_util.py
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python

[[ -f "${HOLDER}" ]] || { echo "missing ${HOLDER}" >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "GPU holder session already exists: ${SESSION}"
  exit 0
fi

mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}'; while [[ ! -f '${REPORT}' ]]; do sleep 10; done; while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; do sleep 10; done; exec '${PYTHON}' -u '${HOLDER}' --devices 0,1,2,3,4,5,6,7 --target-util 60 --target-memory-percent 60 --cycle-seconds 1 --matrix-size 16384 --dtype bfloat16 --sync-every 1 --log-interval 10 >>'${LOG}' 2>&1"
echo "GPU_HOLDER_WATCHER=${SESSION}"
echo "GPU_HOLDER_LOG=${LOG}"
