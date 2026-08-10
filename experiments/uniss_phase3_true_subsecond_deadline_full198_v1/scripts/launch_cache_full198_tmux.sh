#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SESSION="${CACHE_TMUX_SESSION:-uniss_true_subsecond_cache_full198}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "trajectory cache session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd $(printf '%q' "${REPO_ROOT}") && exec bash $(printf '%q' "${SCRIPT_DIR}/run_cache_full198_8gpu.sh")"
echo "started trajectory cache session: ${SESSION}"
