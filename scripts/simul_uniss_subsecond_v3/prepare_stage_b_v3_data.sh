#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "$(dirname "${V3_TRAIN_SELECTION}")" "$(dirname "${V3_MIXED_TRAIN_MANIFEST}")"

python -m training.simul_uniss.subsecond_v3.build_balanced_selection \
  --source-manifest "${SOURCE_TRAIN_MANIFEST}" \
  --output "${V3_TRAIN_SELECTION}" \
  --per-direction "${V3_PER_DIRECTION}" --seed 20260803

python -m training.simul_uniss.subsecond_v3.build_balanced_selection \
  --source-manifest "${SOURCE_VALID_MANIFEST}" \
  --output "${V3_VALID_SELECTION}" --all-records --seed 20260803

SOURCE_MANIFEST="${SOURCE_TRAIN_MANIFEST}" \
SELECTION_MANIFEST="${V3_TRAIN_SELECTION}" \
OUTPUT_ROOT="${V3_PREFIX_TRAIN_ROOT}" WORLD_SIZE=8 \
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/run_prefix_hidden_sidecar.sh"

SOURCE_MANIFEST="${SOURCE_VALID_MANIFEST}" \
SELECTION_MANIFEST="${V3_VALID_SELECTION}" \
OUTPUT_ROOT="${V3_PREFIX_VALID_ROOT}" WORLD_SIZE=8 \
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/run_prefix_hidden_sidecar.sh"

python -m training.simul_uniss.subsecond_v3.build_mixed_manifest \
  --selection-manifest "${V3_TRAIN_SELECTION}" \
  --prefix-manifest "${V3_PREFIX_TRAIN_ROOT}/manifest.jsonl" \
  --clone-manifest "${V2_CLONE_TRAIN_MANIFEST}" \
  --output "${V3_MIXED_TRAIN_MANIFEST}"

python -m training.simul_uniss.subsecond_v3.build_mixed_manifest \
  --selection-manifest "${V3_VALID_SELECTION}" \
  --prefix-manifest "${V3_PREFIX_VALID_ROOT}/manifest.jsonl" \
  --clone-manifest "${V2_CLONE_VALID_MANIFEST}" \
  --output "${V3_MIXED_VALID_MANIFEST}"
