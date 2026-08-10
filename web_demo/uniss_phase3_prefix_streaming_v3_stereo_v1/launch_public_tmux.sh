#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1"
SESSION="${UNISS_PREFIX_STREAMING_TMUX_SESSION:-uniss_phase3_prefix_streaming_v3_demo}"
GPU_ID="${UNISS_PREFIX_STREAMING_GPU:-0}"
PORT="${UNISS_PREFIX_STREAMING_PORT:-7865}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Demo already running: ${SESSION}" >&2
  exit 1
fi
used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ "${used}" -ge "${UNISS_PREFIX_STREAMING_MAX_IDLE_MEMORY_MIB:-2048}" ]]; then
  echo "GPU ${GPU_ID} is not idle: ${used} MiB" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
  echo "Port ${PORT} is already in use" >&2
  exit 1
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU_ID}' UNISS_PREFIX_STREAMING_DEVICE='cuda:0' UNISS_PREFIX_STREAMING_PORT='${PORT}' '${SCRIPT_DIR}/share_watchdog.sh'"
echo "SESSION=${SESSION}"
echo "GPU=${GPU_ID}"
echo "PORT=${PORT}"
echo "AUTH_MODE=public_no_login"

