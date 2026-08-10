#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_PREFIX_LONGFORM_TMUX_SESSION:-uniss_phase3_prefix_longform_v1_demo}"
tmux has-session -t "${SESSION}" 2>/dev/null && echo "TMUX=${SESSION}:running" || echo "TMUX=${SESSION}:not-running"
[[ -f "${SCRIPT_DIR}/access_info.json" ]] && cat "${SCRIPT_DIR}/access_info.json" || echo "Access info not ready"
if [[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]]; then
  tail -n 80 "${SCRIPT_DIR}/runtime_logs/public_server.log"
fi
