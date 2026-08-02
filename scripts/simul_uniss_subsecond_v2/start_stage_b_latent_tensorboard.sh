#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_LATENT_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_latent_formal_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${STAGE_B_LATENT_RUN_ROOT}/tensorboard"
exec tensorboard \
  --logdir "${STAGE_B_LATENT_RUN_ROOT}/tensorboard" \
  --host 0.0.0.0 \
  --port "${STAGE_B_LATENT_TENSORBOARD_PORT}"
