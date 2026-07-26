#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
[[ -n "${STAGE4_TRAIN_ITERS:-}" ]] || { echo "Missing ${TRAINING_SCHEDULE_FILE}" >&2; exit 1; }
exec "${REPO_ROOT}/scripts/simul_uniss/train_qwen_stage.sh" --config "${EXPERIMENT_DIR}/experiment.env" --stage interleaved "$@"
