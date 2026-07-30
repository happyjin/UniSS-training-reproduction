#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1.env"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
mkdir -p "${LOG_DIR}" "${STAGE4_TENSORBOARD_DIR}"
exec tensorboard --logdir "${STAGE4_TENSORBOARD_DIR}" --host 0.0.0.0 \
  --port "${STAGE_D_TENSORBOARD_PORT}" 2>&1 | tee -a "${LOG_DIR}/tensorboard.log"
