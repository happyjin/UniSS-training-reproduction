#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${REPO_ROOT}/scripts/simul_uniss/run_stage5_streaming_replay.sh" --config "${EXPERIMENT_DIR}/experiment.env" --decoder bicodec --record-index 0
  printf '\n'; exit 0
fi
exec "${REPO_ROOT}/scripts/simul_uniss/run_stage5_streaming_replay.sh" --config "${EXPERIMENT_DIR}/experiment.env" --decoder bicodec "$@"
