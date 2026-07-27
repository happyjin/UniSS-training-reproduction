#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/tensorboard"
tmux has-session -t "${TENSORBOARD_SESSION}" 2>/dev/null && {
  echo "TensorBoard session already exists: ${TENSORBOARD_SESSION}"
  exit 0
}
tmux new-session -d -s "${TENSORBOARD_SESSION}" \
  "'${TRAIN_ENV}/bin/tensorboard' --logdir '${RUN_ROOT}/tensorboard' --host 0.0.0.0 --port '${TENSORBOARD_PORT}' 2>&1 | tee '${LOG_ROOT}/tensorboard.log'"
echo "TensorBoard: http://127.0.0.1:${TENSORBOARD_PORT}"

