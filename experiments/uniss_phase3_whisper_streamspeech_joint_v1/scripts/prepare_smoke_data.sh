#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

mkdir -p "${SMOKE_ROOT}" "${REPLAY_INDEX_ROOT}"
"${PYTHON}" "${REPO_ROOT}/training/phase3_whisper_streamspeech_joint/build_joint_manifests.py" \
  --train-source "${PILOT_STAGE_A_ROOT}/parts/train-00000/manifest.jsonl" \
  --train-source "${PILOT_STAGE_A_ROOT}/parts/train-00004/manifest.jsonl" \
  --valid-source "${UNIST_DEV_STAGE_A}" \
  --output-dir "${SMOKE_ROOT}" \
  --phase3-model "${PHASE3_MODEL}" \
  --validation-per-mille 0 \
  --limit 128

"${PYTHON}" -m training.phase3_whisper_streamspeech_joint.build_replay_index \
  --source "${PHASE3_REPLAY_PACKED}" \
  --output "${SMOKE_REPLAY_OFFSETS}" \
  --max-records 8 \
  --progress-interval 0
