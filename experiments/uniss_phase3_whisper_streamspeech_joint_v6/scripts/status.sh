#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
tmux list-sessions 2>/dev/null | grep 'uniss_phase3_joint_v6' || true
find "${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v6" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' | sort -n | tail -n 4
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
