#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

SESSION="${SESSION:-uniss_phase3_joint_replay_index}"
LOG="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/full_replay_index.log"
refuse_existing "${FULL_REPLAY_OFFSETS}" "${FULL_REPLAY_OFFSETS}.json" "${LOG}"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
COMMAND="${PYTHON} -m training.phase3_whisper_streamspeech_joint.build_replay_index --source ${PHASE3_REPLAY_PACKED} --output ${FULL_REPLAY_OFFSETS} --progress-interval 100000 2>&1 | tee ${LOG}"
tmux new-session -d -s "${SESSION}" "cd ${REPO_ROOT} && ${COMMAND}"
echo "Started ${SESSION}; log: ${LOG}"
