#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_PREFIX_STREAMING_TMUX_SESSION:-uniss_phase3_prefix_streaming_v3_demo}"
tmux has-session -t "${SESSION}" 2>/dev/null && echo "TMUX=${SESSION}:running" || echo "TMUX=${SESSION}:not-running"
[[ -f "${SCRIPT_DIR}/access_info.json" ]] && cat "${SCRIPT_DIR}/access_info.json" || echo "Access info not ready"
[[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]] && tail -n 50 "${SCRIPT_DIR}/runtime_logs/public_server.log" || true

