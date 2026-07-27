#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/offline_s2st_phase3_v1"
SESSION="${UNISS_DEMO_TMUX_SESSION:-uniss_offline_phase3_demo}"
GPU_ID="${UNISS_DEMO_GPU:-0}"
PORT="${UNISS_DEMO_PORT:-7861}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Run ${SCRIPT_DIR}/setup_environment.sh first" >&2; exit 1; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Demo tmux session already exists: ${SESSION}" >&2
  exit 1
fi
memory_used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ "${memory_used}" -ge "${UNISS_DEMO_MAX_IDLE_MEMORY_MIB:-1024}" ]]; then
  echo "GPU ${GPU_ID} is not idle: ${memory_used} MiB used" >&2
  exit 1
fi
AUTH_USER="${UNISS_DEMO_AUTH_USER:-uniss}"
AUTH_PASSWORD="${UNISS_DEMO_AUTH_PASSWORD:-$("${DEMO_ENV}/bin/python" -c 'import secrets; print(secrets.token_urlsafe(16))')}"
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU_ID}' UNISS_DEMO_DEVICE='cuda:0' UNISS_DEMO_PORT='${PORT}' UNISS_DEMO_AUTH_USER='${AUTH_USER}' UNISS_DEMO_AUTH_PASSWORD='${AUTH_PASSWORD}' '${SCRIPT_DIR}/run_public.sh'"
echo "SESSION=${SESSION}"
echo "GPU=${GPU_ID}"
echo "AUTH_USER=${AUTH_USER}"
echo "AUTH_PASSWORD=${AUTH_PASSWORD}"
echo "PUBLIC_URL_FILE=${SCRIPT_DIR}/public_url.txt"
echo "ACCESS_INFO_FILE=${SCRIPT_DIR}/access_info.json"
