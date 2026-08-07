#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

STAGE_A_TRACKER="${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/${STAGE_A_RUN_NAME}/latest_checkpointed_iteration.txt"
require_file "${STAGE_A_TRACKER}"
[[ "$(cat "${STAGE_A_TRACKER}")" == "500" ]] || {
  echo "Stage A checkpoint is not complete: ${STAGE_A_TRACKER}" >&2
  exit 1
}
tmux_session_exists "${STAGE_B_SESSION}" && {
  echo "tmux session already exists: ${STAGE_B_SESSION}" >&2
  exit 1
}
tmux new-session -d -s "${STAGE_B_SESSION}" \
  "cd '${REPO_ROOT}' && bash '${SCRIPT_ROOT}/run_stage_b.sh'"
echo "started ${STAGE_B_SESSION}"
