#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${STAGE_C_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_c_source_proxy_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
mkdir -p "${STAGE_C_LOG_DIR}" "${STAGE_C_TENSORBOARD_DIR}"
exec tensorboard \
  --logdir "${STAGE_C_TENSORBOARD_DIR}" \
  --host 0.0.0.0 \
  --port "${STAGE_C_TENSORBOARD_PORT}" \
  2>&1 | tee -a "${STAGE_C_LOG_DIR}/tensorboard.log"
