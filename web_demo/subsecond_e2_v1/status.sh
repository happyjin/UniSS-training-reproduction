#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${UNISS_E2_TMUX_SESSION:-uniss_subsecond_e2_demo}"
LOG="${REPO_ROOT}/web_demo/subsecond_e2_v1/runtime_logs/public_server.log"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "RUNNING=${SESSION}"
else
  echo "STOPPED=${SESSION}"
fi
if [[ -f "${LOG}" ]]; then
  grep -E 'Running on public URL|PUBLIC_URL=' "${LOG}" | tail -1 || true
  tail -20 "${LOG}"
fi
