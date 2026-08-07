#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${PIPELINE_SESSION}" >&2
  exit 1
}
tmux new-session -d -s "${PIPELINE_SESSION}" \
  "cd '${REPO_ROOT}' && bash '${SCRIPT_ROOT}/run_pipeline.sh'"
echo "started ${PIPELINE_SESSION}"
