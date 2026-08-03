#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_C_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_c_after_v3_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

SESSION="${STAGE_C_V3_TENSORBOARD_SESSION:-uniss_stage_c_after_v3_tensorboard}"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
tmux new-session -d -s "${SESSION}" \
  "tensorboard --logdir '${STAGE_C_V3_RUN_DIR}/tensorboard' --host 0.0.0.0 --port '${STAGE_C_V3_TENSORBOARD_PORT}'"
echo "SESSION=${SESSION}"
echo "URL=http://127.0.0.1:${STAGE_C_V3_TENSORBOARD_PORT}"
