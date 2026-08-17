#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

: "${RUN_CURRICULUM_ITERS:?RUN_CURRICULUM_ITERS is required}"

dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=(--dry-run)
  shift
fi

exec bash "${V2_ROOT}/scripts/run_stage_a_megatron.sh" \
  "${dry_run[@]}" \
  --stage-a-curriculum-iters "${RUN_CURRICULUM_ITERS}" \
  "$@"
