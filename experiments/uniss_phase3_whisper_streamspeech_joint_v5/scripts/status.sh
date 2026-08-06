#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

tmux ls 2>/dev/null | grep -E 'uniss_phase3_joint_v5|uniss_phase3_joint_v5_tensorboard' || true
pgrep -af 'phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader
