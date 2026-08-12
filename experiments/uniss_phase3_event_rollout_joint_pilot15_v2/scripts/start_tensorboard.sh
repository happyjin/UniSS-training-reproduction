#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.env"
mkdir -p "${TB_DIR}" "$(dirname "${LOG_PATH}")"
exec "${ENV_ROOT}/bin/tensorboard" --logdir "${TB_DIR}" --host 0.0.0.0 \
  --port "${TENSORBOARD_PORT}" --reload_interval 5
