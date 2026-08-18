#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID=${DATA_RUN_ID:?set DATA_RUN_ID to an immutable formal run ID}
SESSION=${TMUX_SESSION:-uniss_e2e_gold_${RUN_ID}}
LOG=${SCRIPT_DIR}/../../../logs/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/${RUN_ID}/launcher.log
mkdir -p "$(dirname -- "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && DATA_RUN_ID=${RUN_ID} ${SCRIPT_DIR}/run_formal_gold_trajectories.sh 2>&1 | tee ${LOG}"
echo "session=${SESSION}"
echo "log=${LOG}"
