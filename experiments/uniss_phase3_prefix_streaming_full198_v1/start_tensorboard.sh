#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env"
SESSION="${RUN_NAME}_tensorboard"
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${TENSORBOARD_PORT}$"; then
  echo "TensorBoard port ${TENSORBOARD_PORT} is already occupied" >&2
  exit 1
fi
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "${PYTHON} -m tensorboard.main --logdir '${TB_DIR}' --host 0.0.0.0 --port '${TENSORBOARD_PORT}' --load_fast=false"
echo "TensorBoard session=${SESSION} port=${TENSORBOARD_PORT} logdir=${TB_DIR}"

