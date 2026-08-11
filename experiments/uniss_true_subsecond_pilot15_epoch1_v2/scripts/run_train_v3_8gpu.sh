#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

# Keep the fully audited v2 data immutable while isolating every artifact from
# the two preflight failures (OOM and legacy GLM range parsing).
export PACKED_ROOT REPORT_ROOT
export EXPERIMENT_NAME="uniss_true_subsecond_pilot15_epoch1_v3"
export LOG_ROOT="${REPO_ROOT}/logs/${EXPERIMENT_NAME}"
export RUN_ROOT="${REPO_ROOT}/runs/${EXPERIMENT_NAME}"
export SAVE_ROOT="${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}"
export TENSORBOARD_PORT=6072

exec bash "${SCRIPT_DIR}/run_train_8gpu.sh"
