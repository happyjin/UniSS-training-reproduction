#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

RUN_ID=${RUN_ID:-teacher_cache_smoke8_$(date -u +%Y%m%dT%H%M%SZ)}
ROOT="${DATA_ROOT}/stage_a_teacher_cache_smoke/${RUN_ID}"

CACHE_PACKS="${SMOKE_TRAIN_PACKS}" \
CACHE_ROOT="${ROOT}/train" \
CACHE_COVERAGE_EPOCHS=32 \
CACHE_MAX_ACOUSTICS=1 \
CACHE_PROGRESS_INTERVAL=1 \
  bash "${SCRIPT_DIR}/run_teacher_cache_8gpu.sh"

CACHE_PACKS="${SMOKE_VALID_PACKS}" \
CACHE_ROOT="${ROOT}/valid" \
CACHE_COVERAGE_EPOCHS=1 \
CACHE_MAX_ACOUSTICS=1 \
CACHE_PROGRESS_INTERVAL=1 \
  bash "${SCRIPT_DIR}/run_teacher_cache_8gpu.sh"

echo "SMOKE_TEACHER_ROOT=${ROOT}"
