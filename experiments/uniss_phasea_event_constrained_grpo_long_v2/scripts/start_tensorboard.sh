#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
exec "${PYTHON}" -m tensorboard.main \
  --logdir "${REPO_ROOT}/runs/uniss_phasea_event_constrained_grpo_long_v2/tensorboard" \
  --host 0.0.0.0 --port 6019 --reload_interval 5
