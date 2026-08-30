#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
SESSION=${GPU_HOLDER_SESSION:-uniss_gpu_load_60}
LOG=${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1/gpu_holder_after_completion.log
HOLDER=${REPO_ROOT}/scripts/gpu_load/target_gpu_util.py
[[ -f "${HOLDER}" ]] || { echo "missing holder: ${HOLDER}" >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then echo "holder already running"; exit 0; fi
remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l || true)
[[ ${remaining} -eq 0 ]] || { echo "refusing holder while ${remaining} GPU processes remain" >&2; exit 3; }
mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}'; exec '${PYTHON}' -u '${HOLDER}' --devices 0,1,2,3,4,5,6,7 --target-util 60 --target-memory-percent 60 --cycle-seconds 1 --matrix-size 16384 --dtype bfloat16 --sync-every 1 --log-interval 10 >>'${LOG}' 2>&1"
echo "GPU_HOLDER_SESSION=${SESSION}"
