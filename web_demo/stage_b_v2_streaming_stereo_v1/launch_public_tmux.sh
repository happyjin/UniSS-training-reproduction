#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/stage_b_v2_streaming_stereo_v1"
SESSION="${UNISS_STUDENT_V2_DEMO_TMUX_SESSION:-uniss_student_v2_streaming_demo}"
GPU_ID="${UNISS_STUDENT_V2_DEMO_GPU:-0}"
PORT="${UNISS_STUDENT_V2_DEMO_PORT:-7864}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Demo tmux session already exists: ${SESSION}" >&2
  exit 1
fi
memory_used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ "${memory_used}" -ge "${UNISS_STUDENT_V2_MAX_IDLE_MEMORY_MIB:-2048}" ]]; then
  echo "GPU ${GPU_ID} is not idle: ${memory_used} MiB used" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
  echo "Port ${PORT} is already in use" >&2
  exit 1
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU_ID}' UNISS_STUDENT_V2_DEMO_DEVICE='cuda:0' UNISS_STUDENT_V2_DEMO_PORT='${PORT}' '${SCRIPT_DIR}/share_watchdog.sh'"
echo "SESSION=${SESSION}"
echo "GPU=${GPU_ID}"
echo "PORT=${PORT}"
echo "AUTH_MODE=public_no_login"
