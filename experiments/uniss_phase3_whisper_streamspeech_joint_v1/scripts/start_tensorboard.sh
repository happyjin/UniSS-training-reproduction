#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

tmux kill-session -t "${TENSORBOARD_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TENSORBOARD_SESSION}" \
  "${PYTHON} -m tensorboard.main --logdir ${REPO_ROOT}/runs/uniss_phase3_whisper_streamspeech_joint_v1 --host 127.0.0.1 --port ${TENSORBOARD_PORT}"
echo "TensorBoard: http://127.0.0.1:${TENSORBOARD_PORT}/"
