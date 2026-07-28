#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

echo "Waiting for frozen Reward-v2 full-dev comparison: ${DEV_COMPARE_ROOT}/COMPLETE"
while [[ ! -f "${DEV_COMPARE_ROOT}/COMPLETE" ]]; do sleep 30; done
exec "${SCRIPT_DIR}/launch_all_tmux.sh"

