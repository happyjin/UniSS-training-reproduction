#!/usr/bin/env bash
set -euo pipefail
V13_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${V13_DIR}/config_canary.env"
exec "${PYTHON}" -m tensorboard.main --logdir "${TB_DIR}" --host 0.0.0.0 --port "${TENSORBOARD_PORT}"
