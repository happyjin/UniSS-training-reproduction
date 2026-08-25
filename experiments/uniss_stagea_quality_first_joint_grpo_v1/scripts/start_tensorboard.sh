#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
PORT=${1:-6017}
exec "${PYTHON_BIN}" -m tensorboard.main --logdir "${RUN_ROOT}/tensorboard" --host 0.0.0.0 --port "${PORT}" --load_fast=false

