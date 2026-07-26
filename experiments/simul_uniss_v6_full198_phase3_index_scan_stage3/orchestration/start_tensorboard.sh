#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
exec "${REPO_ROOT}/scripts/simul_uniss/start_tensorboard.sh" --config "${EXPERIMENT_DIR}/experiment.env"
