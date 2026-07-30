#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/streaming_s2st_r2_v1"
SESSION="${UNISS_STREAMING_TMUX_SESSION:-uniss_streaming_r2_demo}"
GPU_ID="${UNISS_STREAMING_GPU:-1}"
PORT="${UNISS_STREAMING_PORT:-7862}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Run ${SCRIPT_DIR}/setup_environment.sh first" >&2; exit 1; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Demo tmux session already exists: ${SESSION}" >&2
  exit 1
fi
memory_used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ "${memory_used}" -ge "${UNISS_STREAMING_MAX_IDLE_MEMORY_MIB:-2048}" ]]; then
  echo "GPU ${GPU_ID} is not idle: ${memory_used} MiB used" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
  echo "Port ${PORT} is already in use" >&2
  exit 1
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU_ID}' UNISS_STREAMING_DEVICE='cuda:0' UNISS_STREAMING_PORT='${PORT}' '${SCRIPT_DIR}/share_watchdog.sh'"
echo "SESSION=${SESSION}"
echo "GPU=${GPU_ID}"
echo "PORT=${PORT}"
echo "AUTH_MODE=public_no_login"
echo "PUBLIC_URL_FILE=${SCRIPT_DIR}/public_url.txt"
echo "ACCESS_INFO_FILE=${SCRIPT_DIR}/access_info.json"
