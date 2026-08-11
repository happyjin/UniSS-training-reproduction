#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${PIPELINE_TMUX_SESSION:-uniss_true_subsecond_pilot15_epoch1_v2}"
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}"
  exit 0
}
tmux new-session -d -s "${SESSION}" \
  "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && bash ${SCRIPT_DIR}/run_pipeline.sh"
echo "started ${SESSION}"
