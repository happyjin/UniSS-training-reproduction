#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${STAGE_C_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_c_source_proxy_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

MODE="${1:-formal}"
mkdir -p "${STAGE_C_OUTPUT_DIR}" "${STAGE_C_TENSORBOARD_DIR}" "${STAGE_C_LOG_DIR}"
if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_DIR="${STAGE_C_OUTPUT_DIR}_smoke"
  TENSORBOARD_DIR="${STAGE_C_TENSORBOARD_DIR}_smoke"
  LOG_FILE="${STAGE_C_LOG_DIR}/smoke_train.log"
  TRAIN_ARGS=(--batch-size 4 --max-steps 4 --eval-interval 2 --eval-batches 1 --calibration-batches 1 --save-interval 2 --log-interval 1 --num-workers 0 --max-audio-seconds 2)
else
  OUTPUT_DIR="${STAGE_C_OUTPUT_DIR}"
  TENSORBOARD_DIR="${STAGE_C_TENSORBOARD_DIR}"
  LOG_FILE="${STAGE_C_LOG_DIR}/formal_train.log"
  TRAIN_ARGS=(--batch-size "${STAGE_C_BATCH_SIZE}" --max-steps "${STAGE_C_MAX_STEPS}" --learning-rate "${STAGE_C_LEARNING_RATE}" --num-workers "${STAGE_C_NUM_WORKERS}" --eval-interval 500 --eval-batches 4 --calibration-batches 32 --save-interval 500 --log-interval 10)
fi

RESUME_ARGS=()
if [[ "${STAGE_C_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

"${CONDA_PREFIX}/bin/torchrun" \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 8 \
  --master-addr 127.0.0.1 \
  --master-port "${STAGE_C_MASTER_PORT}" \
  -m training.simul_uniss.subsecond_v1.train_stage_c \
  --train-manifest "${STAGE_C_TRAIN_MANIFEST}" \
  --valid-manifest "${STAGE_C_VALID_MANIFEST}" \
  --student-checkpoint "${STAGE_C_STUDENT_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --device cuda \
  --bf16 \
  "${TRAIN_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
