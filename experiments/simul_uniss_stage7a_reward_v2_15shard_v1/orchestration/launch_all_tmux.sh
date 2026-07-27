#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "At least one GPU already has a compute process; refusing mixed launch" >&2
  exit 1
fi

launch() {
  local session="$1"
  shift
  tmux has-session -t "${session}" 2>/dev/null && {
    echo "tmux session already exists: ${session}" >&2
    exit 1
  }
  tmux new-session -d -s "${session}" "cd '${REPO_ROOT}' && exec $*"
}

launch stage7a_reward_v2_r0 "${ROOT}/r0_bias_sweep/run_2gpu.sh"
launch stage7a_reward_v2_r1 "${ROOT}/common/run_train_2gpu.sh r1"
launch stage7a_reward_v2_r2 "${ROOT}/common/run_train_2gpu.sh r2"
launch stage7a_reward_v2_r3 "${ROOT}/common/run_train_2gpu.sh r3"
launch stage7a_reward_v2_compare "${ROOT}/evaluation/wait_and_compare.sh"

tmux ls | grep 'stage7a_reward_v2_'
