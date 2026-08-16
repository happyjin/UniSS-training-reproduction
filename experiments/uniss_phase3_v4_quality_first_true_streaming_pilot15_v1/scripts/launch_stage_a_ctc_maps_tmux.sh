#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="${EXPERIMENT_NAME}_stage_a_ctc_maps"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${STAGE_A_CTC_MAP_ROOT}" ]]; then
  echo "refusing to overwrite Stage A CTC map root: ${STAGE_A_CTC_MAP_ROOT}" >&2
  exit 3
fi
LOG="${LOG_ROOT}/stage_a/ctc_map_build_v2_launch.log"
mkdir -p "$(dirname "${LOG}")"
COMMAND="cd '${REPO_ROOT}' && bash '${EXPERIMENT_DIR}/scripts/run_stage_a_build_ctc_maps.sh' 2>&1 | tee '${LOG}'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${COMMAND}\""
echo "session=${SESSION}"
echo "log=${LOG}"
