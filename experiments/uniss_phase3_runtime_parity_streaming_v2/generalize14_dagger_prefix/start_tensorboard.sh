#!/usr/bin/env bash
set -euo pipefail

V14_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${V14_DIR}/config_canary.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

exec "${PYTHON}" -m tensorboard.main \
  --logdir "${TB_DIR}" --host 0.0.0.0 --port "${TENSORBOARD_PORT}"
