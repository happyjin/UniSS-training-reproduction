#!/usr/bin/env bash
set -euo pipefail

SESSION="${UNISS_DEMO_TMUX_SESSION:-uniss_offline_phase3_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped ${SESSION}"
else
  echo "Session not running: ${SESSION}"
fi
