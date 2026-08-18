#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"

FORMAL_RUN_ID=${FORMAL_RUN_ID:-v1_rollout_formal_$(date -u +%Y%m%dT%H%M%SZ)}
NUM_GPUS=${NUM_GPUS:-8}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-24}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

DATA_RUN_ID=${DATA_RUN_ID} \
ROLLOUT_RUN_ID=${FORMAL_RUN_ID}_train \
ROLLOUT_SPLIT=train \
ROLLOUT_START_INDEX=0 \
ROLLOUT_LIMIT=0 \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_v1_rollout_8gpu.sh"

DATA_RUN_ID=${DATA_RUN_ID} \
ROLLOUT_RUN_ID=${FORMAL_RUN_ID}_valid \
ROLLOUT_SPLIT=valid \
ROLLOUT_START_INDEX=0 \
ROLLOUT_LIMIT=0 \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_v1_rollout_8gpu.sh"

echo "formal_run_id=${FORMAL_RUN_ID}"
echo "train_report=${REPORT_ROOT}/v1_rollouts/${FORMAL_RUN_ID}_train/AUDIT.json"
echo "valid_report=${REPORT_ROOT}/v1_rollouts/${FORMAL_RUN_ID}_valid/AUDIT.json"
