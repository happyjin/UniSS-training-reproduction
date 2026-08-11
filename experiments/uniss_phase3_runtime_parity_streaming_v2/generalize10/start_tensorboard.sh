#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/generalize10/config.env"
exec "${ENV_ROOT}/bin/tensorboard" --logdir "${TB_DIR}" --host 0.0.0.0 \
  --port "${TENSORBOARD_PORT}" --reload_interval 5
