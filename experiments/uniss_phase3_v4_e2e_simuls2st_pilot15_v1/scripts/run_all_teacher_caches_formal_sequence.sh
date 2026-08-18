#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"

FORMAL_RUN_ID=${FORMAL_RUN_ID:-teacher_cache_formal_$(date -u +%Y%m%dT%H%M%SZ)}
NUM_GPUS=${NUM_GPUS:-8}
PHASE3_PROCESSES_PER_GPU=${PHASE3_PROCESSES_PER_GPU:-1}
V1_PROCESSES_PER_GPU=${V1_PROCESSES_PER_GPU:-1}
PHASE3_SMOKE_LIMIT=${PHASE3_SMOKE_LIMIT:-256}
V1_SMOKE_LIMIT=${V1_SMOKE_LIMIT:-64}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

DATA_RUN_ID=${DATA_RUN_ID} \
V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_valid \
TEACHER_RUN_ID=${FORMAL_RUN_ID}_phase3_smoke \
TEACHER_SPLIT=valid \
TEACHER_LIMIT=${PHASE3_SMOKE_LIMIT} \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${PHASE3_PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"

DATA_RUN_ID=${DATA_RUN_ID} \
V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_valid \
TEACHER_RUN_ID=${FORMAL_RUN_ID}_v1_smoke \
TEACHER_SPLIT=valid \
TEACHER_LIMIT=${V1_SMOKE_LIMIT} \
NUM_GPUS=${NUM_GPUS} \
PROCESSES_PER_GPU=${V1_PROCESSES_PER_GPU} \
"${SCRIPT_DIR}/run_v1_teacher_cache_8gpu.sh"

for split in train valid; do
  DATA_RUN_ID=${DATA_RUN_ID} \
  V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_${split} \
  TEACHER_RUN_ID=${FORMAL_RUN_ID}_phase3_${split} \
  TEACHER_SPLIT=${split} \
  TEACHER_LIMIT=0 \
  NUM_GPUS=${NUM_GPUS} \
  PROCESSES_PER_GPU=${PHASE3_PROCESSES_PER_GPU} \
  "${SCRIPT_DIR}/run_phase3_teacher_cache_8gpu.sh"
done

for split in train valid; do
  DATA_RUN_ID=${DATA_RUN_ID} \
  V1_ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_${split} \
  TEACHER_RUN_ID=${FORMAL_RUN_ID}_v1_${split} \
  TEACHER_SPLIT=${split} \
  TEACHER_LIMIT=0 \
  NUM_GPUS=${NUM_GPUS} \
  PROCESSES_PER_GPU=${V1_PROCESSES_PER_GPU} \
  "${SCRIPT_DIR}/run_v1_teacher_cache_8gpu.sh"
done

echo "formal_run_id=${FORMAL_RUN_ID}"
echo "phase3_train_audit=${REPORT_ROOT}/phase3_teacher_cache/${FORMAL_RUN_ID}_phase3_train/AUDIT.json"
echo "phase3_valid_audit=${REPORT_ROOT}/phase3_teacher_cache/${FORMAL_RUN_ID}_phase3_valid/AUDIT.json"
echo "v1_train_audit=${REPORT_ROOT}/v1_asr_teacher_cache/${FORMAL_RUN_ID}_v1_train/AUDIT.json"
echo "v1_valid_audit=${REPORT_ROOT}/v1_asr_teacher_cache/${FORMAL_RUN_ID}_v1_valid/AUDIT.json"
