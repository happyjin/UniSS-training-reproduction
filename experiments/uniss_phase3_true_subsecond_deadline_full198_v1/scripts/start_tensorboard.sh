#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"
SESSION="${TB_TMUX_SESSION:-uniss_true_subsecond_tensorboard_6070}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TensorBoard already running: ${SESSION}"
  exit 0
fi
mkdir -p "${TB_DIR}"
tmux new-session -d -s "${SESSION}" \
  "${ENV_ROOT}/bin/tensorboard --logdir $(printf '%q' "${TB_DIR}") --host 0.0.0.0 --port $(printf '%q' "${TENSORBOARD_PORT}")"
echo "TensorBoard started on 0.0.0.0:${TENSORBOARD_PORT}; logdir=${TB_DIR}"
