#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_C_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_c_formal_15shard_v1.env}"
MODE="${1:-formal}"
[[ "${MODE}" == "formal" || "${MODE}" == "smoke" ]] || { echo "mode must be formal or smoke" >&2; exit 2; }
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

if [[ "${MODE}" == "smoke" ]]; then
  train_manifest="${STAGE_C_SMOKE_TRAIN_MANIFEST}"
  valid_manifest="${STAGE_C_SMOKE_VALID_MANIFEST}"
  student_checkpoint="${STAGE_C_SMOKE_STUDENT_CHECKPOINT}"
  output_dir="${STAGE_C_SMOKE_OUTPUT_DIR}"
  tensorboard_dir="${STAGE_C_SMOKE_TENSORBOARD_DIR}"
  nproc=1
  train_args=(--batch-size 4 --max-steps 2 --eval-interval 1 --eval-batches 1 --calibration-batches 1 --save-interval 1 --log-interval 1 --num-workers 0 --max-audio-seconds 2)
else
  train_manifest="${STAGE_C_TRAIN_MANIFEST}"
  valid_manifest="${STAGE_C_VALID_MANIFEST}"
  student_checkpoint="${STAGE_C_STUDENT_CHECKPOINT}"
  output_dir="${STAGE_C_OUTPUT_DIR}"
  tensorboard_dir="${STAGE_C_TENSORBOARD_DIR}"
  nproc=8
  train_args=(--batch-size "${STAGE_C_BATCH_SIZE}" --max-steps "${STAGE_C_MAX_STEPS}" --learning-rate "${STAGE_C_LEARNING_RATE}" --num-workers "${STAGE_C_NUM_WORKERS}" --eval-interval 500 --eval-batches 4 --calibration-batches 32 --save-interval 500 --log-interval 10)
fi
for path in "${train_manifest}" "${train_manifest}.offsets.bin" "${valid_manifest}" "${valid_manifest}.offsets.bin" "${student_checkpoint}"; do
  [[ -f "${path}" ]] || { echo "Missing required Stage-C input: ${path}" >&2; exit 1; }
done
mkdir -p "${output_dir}" "${tensorboard_dir}" "${STAGE_C_LOG_DIR}"
resume_args=()
[[ "${STAGE_C_RESUME:-0}" == "1" ]] && resume_args+=(--resume)
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
"${CONDA_PREFIX}/bin/torchrun" \
  --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}" \
  --master-addr 127.0.0.1 --master-port "${STAGE_C_MASTER_PORT}" \
  -m training.simul_uniss.subsecond_v1.train_stage_c \
  --train-manifest "${train_manifest}" \
  --valid-manifest "${valid_manifest}" \
  --student-checkpoint "${student_checkpoint}" \
  --output-dir "${output_dir}" \
  --tensorboard-dir "${tensorboard_dir}" \
  --device cuda --bf16 --formal-target-support \
  "${train_args[@]}" "${resume_args[@]}" \
  2>&1 | tee -a "${STAGE_C_LOG_DIR}/${MODE}_train.log"

