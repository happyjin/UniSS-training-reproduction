#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

RUN_ID=${RUN_ID:-teacher_cache_formal8_$(date -u +%Y%m%dT%H%M%SZ)}
ROOT="${DATA_ROOT}/stage_a_teacher_cache_formal/${RUN_ID}"

CACHE_PACKS="${FORMAL_TRAIN_PACKS}" \
CACHE_ROOT="${ROOT}/train" \
CACHE_COVERAGE_EPOCHS=3 \
CACHE_MAX_ACOUSTICS=2 \
CACHE_PROGRESS_INTERVAL=100 \
  bash "${SCRIPT_DIR}/run_teacher_cache_8gpu.sh"

# Validation is deterministic epoch zero; it does not rotate through the three
# train coverage epochs, but it uses the same two-acoustic cap.
CACHE_PACKS="${FORMAL_VALID_PACKS}" \
CACHE_ROOT="${ROOT}/valid" \
CACHE_COVERAGE_EPOCHS=1 \
CACHE_MAX_ACOUSTICS=2 \
CACHE_PROGRESS_INTERVAL=25 \
  bash "${SCRIPT_DIR}/run_teacher_cache_8gpu.sh"

echo "FORMAL_TEACHER_ROOT=${ROOT}"
