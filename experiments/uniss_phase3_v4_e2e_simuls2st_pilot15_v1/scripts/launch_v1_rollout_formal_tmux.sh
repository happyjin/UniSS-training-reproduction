#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

FORMAL_RUN_ID=${FORMAL_RUN_ID:-v1_rollout_formal_$(date -u +%Y%m%dT%H%M%SZ)}
SESSION=${SESSION:-uniss_e2e_v1_rollout_formal}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

command="cd ${REPO_ROOT} && DATA_RUN_ID=${DATA_RUN_ID} FORMAL_RUN_ID=${FORMAL_RUN_ID} NUM_GPUS=${NUM_GPUS:-8} PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-24} QUALITY_AUDIT_WORKERS=${QUALITY_AUDIT_WORKERS:-64} ${SCRIPT_DIR}/run_v1_rollout_formal_sequence.sh"
tmux new-session -d -s "${SESSION}" "bash -lc '${command}'"
echo "session=${SESSION}"
echo "formal_run_id=${FORMAL_RUN_ID}"
