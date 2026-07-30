#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_STREAMING_TMUX_SESSION:-uniss_streaming_r2_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped ${SESSION}"
else
  echo "${SESSION} is not running"
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
