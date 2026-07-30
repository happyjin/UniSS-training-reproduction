#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_STREAMING_TMUX_SESSION:-uniss_streaming_r2_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
  echo "Stopped tmux session ${SESSION}"
else
  echo "${SESSION} is not running"
fi

mapfile -t app_pids < <(
  pgrep -f -- "python .* -m web_demo\.streaming_s2st_r2_v1\.app_gradio" || true
)
if (( ${#app_pids[@]} )); then
  kill -TERM "${app_pids[@]}" 2>/dev/null || true
  for _ in {1..15}; do
    remaining=0
    for pid in "${app_pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && remaining=1
    done
    (( remaining == 0 )) && break
    sleep 1
  done
  for pid in "${app_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  echo "Stopped isolated streaming app PID(s): ${app_pids[*]}"
fi
rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
