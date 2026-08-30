#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
WATCHER=${GPU_HOLDER_WATCHER_SESSION:-uniss_content_first_holder_recovery_v1}
HOLDER_SESSION=${GPU_HOLDER_SESSION:-uniss_gpu_load_60}
LOG=${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1/gpu_holder_recovery.log

if tmux has-session -t "${HOLDER_SESSION}" 2>/dev/null; then
  echo "GPU holder already running"
  exit 0
fi
if tmux has-session -t "${WATCHER}" 2>/dev/null; then
  echo "GPU holder recovery watcher already running"
  exit 0
fi
mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${WATCHER}" \
  "cd '${REPO_ROOT}'; while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do sleep 10; done; exec bash '${HERE}/scripts/start_gpu_holder.sh' >>'${LOG}' 2>&1"
echo "GPU_HOLDER_RECOVERY_WATCHER=${WATCHER}"
