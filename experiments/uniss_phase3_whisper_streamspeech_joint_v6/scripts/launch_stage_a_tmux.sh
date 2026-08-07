#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
SESSION="${SESSION:-uniss_phase3_joint_v6_stage_a}"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
tmux new-session -d -s "${SESSION}" "cd '${REPO_ROOT}' && bash '${SCRIPT_ROOT}/run_stage_a_15shard.sh'"
echo "started ${SESSION}"
