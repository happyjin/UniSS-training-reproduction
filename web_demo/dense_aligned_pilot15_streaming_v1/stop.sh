#!/usr/bin/env bash
set -euo pipefail

SESSION="${UNISS_DENSE_STREAMING_TMUX_SESSION:-uniss_dense_aligned_streaming_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped ${SESSION}"
else
  echo "Demo is not running: ${SESSION}"
fi

