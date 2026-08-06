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

"${PYTHON}" -m training.phase3_whisper_streamspeech_joint.build_joint_manifests \
  --train-source "${FORMAL_STAGE_A_ROOT}/stage_a_source_manifest.jsonl" \
  --valid-source "${UNIST_DEV_STAGE_A}" \
  --output-dir "${FORMAL_JOINT_ROOT}" \
  --phase3-model "${PHASE3_MODEL}" \
  --validation-per-mille 0
