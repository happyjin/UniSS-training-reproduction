#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
if tmux has-session -t "${TENSORBOARD_SESSION}" 2>/dev/null; then
  echo "TensorBoard session already exists: ${TENSORBOARD_SESSION}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/tensorboard" "${LOG_ROOT}"
tmux new-session -d -s "${TENSORBOARD_SESSION}" \
  "'${TRAIN_ENV}/bin/tensorboard' --logdir '${RUN_ROOT}/tensorboard' --host 0.0.0.0 --port '${TENSORBOARD_PORT}' 2>&1 | tee '${LOG_ROOT}/tensorboard.log'"
echo "http://localhost:${TENSORBOARD_PORT}"
