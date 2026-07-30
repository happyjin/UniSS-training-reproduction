#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_ab.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${STAGE_B_RUN_ROOT}/tensorboard" "${STAGE_B_SMOKE_RUN_ROOT}/tensorboard" "${LOG_ROOT}/stage_b"
tensorboard \
  --logdir_spec "pilot:${STAGE_B_RUN_ROOT}/tensorboard,smoke:${STAGE_B_SMOKE_RUN_ROOT}/tensorboard" \
  --host 0.0.0.0 \
  --port "${TENSORBOARD_PORT}" \
  2>&1 | tee -a "${LOG_ROOT}/stage_b/tensorboard.log"
