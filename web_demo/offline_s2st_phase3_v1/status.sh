#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_DEMO_TMUX_SESSION:-uniss_offline_phase3_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX=${SESSION}:running"
else
  echo "TMUX=${SESSION}:not-running"
fi
if [[ -f "${SCRIPT_DIR}/access_info.json" ]]; then
  cat "${SCRIPT_DIR}/access_info.json"
else
  echo "Access info not ready"
fi
if [[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]]; then
  tail -n 30 "${SCRIPT_DIR}/runtime_logs/public_server.log"
fi
