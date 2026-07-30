#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UNISS_STREAMING_TMUX_SESSION:-uniss_streaming_r2_demo}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX=${SESSION}:running"
else
  echo "TMUX=${SESSION}:not-running"
fi
if [[ -f "${SCRIPT_DIR}/access_info.json" ]]; then
  cat "${SCRIPT_DIR}/access_info.json"
  public_url="$(jq -r '.public_url // empty' "${SCRIPT_DIR}/access_info.json")"
  if [[ -n "${public_url}" ]]; then
    if curl -fsSIL --max-time 15 "${public_url}" >/dev/null; then
      echo "PUBLIC_HEALTH=ok"
    else
      echo "PUBLIC_HEALTH=unreachable"
    fi
  fi
else
  echo "Access info not ready"
fi
if [[ -f "${SCRIPT_DIR}/runtime_logs/public_server.log" ]]; then
  awk '
    /Loading GLM4 tokenizer from:/ { current = "" }
    { current = current $0 ORS }
    END { printf "%s", current }
  ' "${SCRIPT_DIR}/runtime_logs/public_server.log" | tail -n 40
fi
