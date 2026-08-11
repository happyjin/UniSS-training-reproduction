#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="uniss_true_subsecond_pilot15_epoch1_v3_train"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 0
fi
tmux new-session -d -s "${SESSION}" \
  "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && bash ${SCRIPT_DIR}/run_train_v3_8gpu.sh"
echo "started ${SESSION}"
