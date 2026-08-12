#!/usr/bin/env bash
set -euo pipefail

V15_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${V15_DIR}/config_canary.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

exec "${ENV_ROOT}/bin/tensorboard" --logdir "${TB_DIR}" --host 0.0.0.0 --port "${TENSORBOARD_PORT}"

