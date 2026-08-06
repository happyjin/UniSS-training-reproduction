#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

SESSION="${SESSION:-uniss_phase3_joint_full198_stage_a}"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${FORMAL_STAGE_A_ROOT}/parts" "${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/full198_stage_a"
tmux new-session -d -s "${SESSION}" -n lane0
for lane in 0 1 2 3 4 5 6 7; do
  (( lane == 0 )) || tmux new-window -t "${SESSION}" -n "lane${lane}"
  log="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/full198_stage_a/lane${lane}.log"
  command="cd ${REPO_ROOT} && bash ${SCRIPT_ROOT}/prepare_full198_stage_a_worker.sh ${lane} 2>&1 | tee ${log}"
  tmux send-keys -t "${SESSION}:lane${lane}" "${command}" C-m
done
echo "Started ${SESSION}; one window per GPU lane. Existing completed shard markers are resumed safely."
