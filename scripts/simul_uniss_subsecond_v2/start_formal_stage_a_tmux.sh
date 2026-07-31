#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/dlami/nvme/jasonleeeli/projects/UniSS}"
CONFIG="${1:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/formal_15shard.env}"
SESSION="${FORMAL_STAGE_A_TMUX_SESSION:-uniss_formal_stage_a_v2}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 0
fi
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && bash scripts/simul_uniss_subsecond_v2/run_formal_stage_a_15shard.sh '${CONFIG}' all"
echo "started tmux session: ${SESSION}"

