#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

: "${RUN_CURRICULUM_ITERS:?RUN_CURRICULUM_ITERS is required}"
: "${RUN_OPTIMIZER_ITERS:?RUN_OPTIMIZER_ITERS is required}"
: "${RUN_OPTIMIZER_WARMUP_ITERS:?RUN_OPTIMIZER_WARMUP_ITERS is required}"

dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=(--dry-run)
  shift
fi

prefix_schedule=()
if [[ "${RUN_PREFIX_SCHEDULE:-0}" == "1" ]]; then
  prefix_schedule=(--stage-a-prefix-schedule)
fi

exec bash "${V2_ROOT}/scripts/run_stage_a_megatron.sh" \
  "${dry_run[@]}" \
  --stage-a-curriculum-iters "${RUN_CURRICULUM_ITERS}" \
  --stage-a-optimizer-iters "${RUN_OPTIMIZER_ITERS}" \
  --stage-a-optimizer-warmup-iters "${RUN_OPTIMIZER_WARMUP_ITERS}" \
  --lr-decay-iters "${RUN_OPTIMIZER_ITERS}" \
  --lr-warmup-iters "${RUN_OPTIMIZER_WARMUP_ITERS}" \
  "${prefix_schedule[@]}" \
  "$@"
