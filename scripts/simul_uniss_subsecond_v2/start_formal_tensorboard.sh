#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: $0 stage_b|stage_c|stage_d}"
case "${STAGE}" in
  stage_b)
    source "${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_formal_15shard_v1.env"
    port="${STAGE_B_TENSORBOARD_PORT}"
    logdir="${STAGE_B_RUN_ROOT}/tensorboard"
    ;;
  stage_c)
    source "${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_c_formal_15shard_v1.env"
    port="${STAGE_C_TENSORBOARD_PORT}"
    logdir="${STAGE_C_TENSORBOARD_DIR}"
    ;;
  stage_d)
    source "${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_d_formal_15shard_v1.env"
    port="${STAGE_D_TENSORBOARD_PORT}"
    logdir="${STAGE4_TENSORBOARD_DIR}"
    ;;
  *) echo "stage must be stage_b, stage_c, or stage_d" >&2; exit 2 ;;
esac
source "${ACTIVATE_SCRIPT}"
exec tensorboard --logdir "${logdir}" --bind_all --port "${port}"

