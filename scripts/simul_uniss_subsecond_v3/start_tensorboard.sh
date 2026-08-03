#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

SESSION="${STAGE_B_V3_TENSORBOARD_SESSION:-uniss_stage_b_v3_tensorboard}"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
tmux new-session -d -s "${SESSION}" \
  "tensorboard --logdir '${V3_RUN_ROOT}/tensorboard' --host 0.0.0.0 --port '${V3_TENSORBOARD_PORT}'"
echo "SESSION=${SESSION}"
echo "URL=http://127.0.0.1:${V3_TENSORBOARD_PORT}"
