#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/subsecond_e2_v1"
SESSION="${UNISS_E2_TMUX_SESSION:-uniss_subsecond_e2_demo}"
LOG="${SCRIPT_DIR}/runtime_logs/public_server.log"
mkdir -p "${SCRIPT_DIR}/runtime_logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 0
fi
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/run_public.sh' 2>&1 | tee -a '${LOG}'"
echo "TMUX_SESSION=${SESSION}"
echo "LOG=${LOG}"
