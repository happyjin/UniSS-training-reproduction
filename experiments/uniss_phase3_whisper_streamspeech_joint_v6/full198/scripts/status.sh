#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null && echo "tmux: running (${PIPELINE_SESSION})" || echo "tmux: not running"
[[ -f "${FULL198_STATUS_ROOT}/status.txt" ]] && { printf 'pipeline: '; cat "${FULL198_STATUS_ROOT}/status.txt"; }
for run in "${STAGE_A_RUN_NAME}" "${STAGE_B_RUN_NAME}"; do
  tracker="${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/${run}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] && echo "${run}: checkpoint iteration $(cat "${tracker}")"
done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader
