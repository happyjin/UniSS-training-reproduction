#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_PREFIX_STREAMING_TMUX_SESSION:-uniss_phase3_prefix_streaming_v3_demo}"
tmux has-session -t "${SESSION}" 2>/dev/null && echo "TMUX=${SESSION}:running" || echo "TMUX=${SESSION}:not-running"
[[ -f "${SCRIPT_DIR}/access_info.json" ]] && cat "${SCRIPT_DIR}/access_info.json" || echo "Access info not ready"
if [[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]]; then
  awk '
    /Loading GLM4 tokenizer from:/ { current = "" }
    { current = current $0 ORS }
    END { printf "%s", current }
  ' "${SCRIPT_DIR}/runtime_logs/public_server.log" | tail -n 50
fi
