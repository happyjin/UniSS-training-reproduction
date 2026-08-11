#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"
mkdir -p "${TB_DIR}" "${REPO_ROOT}/logs"
exec "${ENV_ROOT}/bin/tensorboard" \
  --logdir "${TB_DIR}" --host 0.0.0.0 --port "${TENSORBOARD_PORT}" \
  --reload_interval 5
