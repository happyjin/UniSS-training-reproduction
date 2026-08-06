#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

SESSION="${SESSION:-uniss_phase3_joint_full198_pipeline}"
LOG="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/full198_pipeline.log"
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session exists: ${SESSION}" >&2
  exit 1
}
refuse_existing "${LOG}"
tmux new-session -d -s "${SESSION}" \
  "cd ${REPO_ROOT} && bash ${SCRIPT_ROOT}/wait_and_train_full198.sh 2>&1 | tee ${LOG}"
echo "Started ${SESSION}; log: ${LOG}"
