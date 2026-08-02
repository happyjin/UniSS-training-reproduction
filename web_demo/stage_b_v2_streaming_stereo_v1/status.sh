#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_STUDENT_V2_DEMO_TMUX_SESSION:-uniss_student_v2_streaming_demo}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX=${SESSION}:running"
else
  echo "TMUX=${SESSION}:not-running"
fi
if [[ -f "${SCRIPT_DIR}/access_info.json" ]]; then
  jq . "${SCRIPT_DIR}/access_info.json"
  public_url="$(jq -r '.public_url // empty' "${SCRIPT_DIR}/access_info.json")"
  if [[ -n "${public_url}" ]] && curl -fsSIL --max-time 15 "${public_url}" >/dev/null; then
    echo "PUBLIC_HEALTH=ok"
  fi
else
  echo "Access info not ready"
fi
if [[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]]; then
  tail -n 50 "${SCRIPT_DIR}/runtime_logs/public_server.log"
fi
