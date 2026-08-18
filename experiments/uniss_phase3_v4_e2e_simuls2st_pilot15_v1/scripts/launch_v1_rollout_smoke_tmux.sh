#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

ROLLOUT_RUN_ID=${ROLLOUT_RUN_ID:-v1_rollout_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_v1_rollout_smoke}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} ROLLOUT_RUN_ID=${ROLLOUT_RUN_ID} ROLLOUT_SPLIT=${ROLLOUT_SPLIT:-valid} ROLLOUT_LIMIT=${ROLLOUT_LIMIT:-32} NUM_GPUS=${NUM_GPUS:-8} PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-1} ${SCRIPT_DIR}/run_v1_rollout_8gpu.sh"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "data_run_id=${DATA_RUN_ID}"
echo "rollout_run_id=${ROLLOUT_RUN_ID}"
