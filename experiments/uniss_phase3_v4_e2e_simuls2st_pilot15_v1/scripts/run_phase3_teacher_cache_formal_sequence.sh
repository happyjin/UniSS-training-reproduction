#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"

FORMAL_RUN_ID=${FORMAL_RUN_ID:-phase3_teacher_formal_$(date -u +%Y%m%dT%H%M%SZ)}
NUM_GPUS=${NUM_GPUS:-8}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-1}
SMOKE_LIMIT=${SMOKE_LIMIT:-256}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

DATA_RUN_ID=${DATA_RUN_ID} \
V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_valid \
TEACHER_RUN_ID=${FORMAL_RUN_ID}_smoke \
TEACHER_SPLIT=valid \
TEACHER_START_INDEX=0 \
TEACHER_LIMIT=${SMOKE_LIMIT} \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"

DATA_RUN_ID=${DATA_RUN_ID} \
V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_train \
TEACHER_RUN_ID=${FORMAL_RUN_ID}_train \
TEACHER_SPLIT=train \
TEACHER_START_INDEX=0 \
TEACHER_LIMIT=0 \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"

DATA_RUN_ID=${DATA_RUN_ID} \
V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_valid \
TEACHER_RUN_ID=${FORMAL_RUN_ID}_valid \
TEACHER_SPLIT=valid \
TEACHER_START_INDEX=0 \
TEACHER_LIMIT=0 \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"

echo "formal_run_id=${FORMAL_RUN_ID}"
echo "train_audit=${REPORT_ROOT}/phase3_teacher_cache/${FORMAL_RUN_ID}_train/AUDIT.json"
echo "valid_audit=${REPORT_ROOT}/phase3_teacher_cache/${FORMAL_RUN_ID}_valid/AUDIT.json"
