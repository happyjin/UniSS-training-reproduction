#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_DENSE_STREAMING_TMUX_SESSION:-uniss_dense_aligned_streaming_demo}"
tmux has-session -t "${SESSION}" 2>/dev/null && echo "SESSION=running" || echo "SESSION=stopped"
[[ -s "${SCRIPT_DIR}/public_url.txt" ]] && echo "PUBLIC_URL=$(tr -d '\r\n' < "${SCRIPT_DIR}/public_url.txt")"
[[ -s "${SCRIPT_DIR}/access_info.json" ]] && echo "ACCESS_INFO=${SCRIPT_DIR}/access_info.json"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader

