#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_STUDENT_V2_DEMO_TMUX_SESSION:-uniss_student_v2_streaming_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped tmux session ${SESSION}"
fi
mapfile -t app_pids < <(
  pgrep -f -- "python .* -m web_demo\.stage_b_v2_streaming_stereo_v1\.app_gradio" || true
)
if (( ${#app_pids[@]} )); then
  kill -TERM "${app_pids[@]}" 2>/dev/null || true
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
