#!/usr/bin/env bash
set -euo pipefail
SESSION=${UNISS_STAGE13_SESSION:-uniss_streamspeech_stage13_public_v1}
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
