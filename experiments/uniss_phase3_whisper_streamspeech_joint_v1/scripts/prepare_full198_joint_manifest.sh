#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

"${PYTHON}" -m training.simul_uniss.subsecond_v1.stage_a assemble \
  --parts-root "${FORMAL_STAGE_A_ROOT}/parts" \
  --output-dir "${FORMAL_STAGE_A_ROOT}" \
  --shard-start 0 \
  --shard-count 198

"${PYTHON}" -m training.simul_uniss.subsecond_v1.stage_a validate \
  --output-dir "${FORMAL_STAGE_A_ROOT}" \
  --samples 64

JOINT_MANIFEST_WORKERS="${JOINT_MANIFEST_WORKERS:-16}"
mapfile -t TRAIN_PARTS < <(
  find "${FORMAL_STAGE_A_ROOT}/parts" -mindepth 2 -maxdepth 2 \
    -name manifest.jsonl -print | sort
)
if (( ${#TRAIN_PARTS[@]} != 198 )); then
  echo "Expected 198 Stage-A part manifests, found ${#TRAIN_PARTS[@]}" >&2
  exit 1
fi

TRAIN_ARGS=()
for manifest in "${TRAIN_PARTS[@]}"; do
  TRAIN_ARGS+=(--train-source "${manifest}")
done

"${PYTHON}" -m training.phase3_whisper_streamspeech_joint.build_joint_manifests_parallel \
  "${TRAIN_ARGS[@]}" \
  --valid-source "${UNIST_DEV_STAGE_A}" \
  --output-dir "${FORMAL_JOINT_ROOT}" \
  --phase3-model "${PHASE3_MODEL}" \
  --workers "${JOINT_MANIFEST_WORKERS}" \
  --validation-per-mille 0 \
  --skip-audio-check \
  --skip-empty-target-bicodec
