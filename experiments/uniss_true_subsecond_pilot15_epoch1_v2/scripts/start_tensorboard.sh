#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"
exec "${ENV_ROOT}/bin/tensorboard" \
  --logdir "${RUN_ROOT}/tensorboard" \
  --host 0.0.0.0 \
  --port "${TENSORBOARD_PORT}"
