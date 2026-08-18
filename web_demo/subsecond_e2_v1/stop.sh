#!/usr/bin/env bash
set -euo pipefail
SESSION="${UNISS_E2_TMUX_SESSION:-uniss_subsecond_e2_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped ${SESSION}"
else
  echo "Not running: ${SESSION}"
fi
