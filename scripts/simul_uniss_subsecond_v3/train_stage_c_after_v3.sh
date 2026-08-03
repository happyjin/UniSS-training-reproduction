#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_C_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_c_after_v3_15shard_v1.env}"
MODE="${1:-formal}"
[[ "${MODE}" == "smoke" || "${MODE}" == "throughput" || "${MODE}" == "formal" ]] || {
  echo "mode must be smoke, throughput, or formal" >&2
  exit 2
}
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${STAGE_C_V3_LOG_DIR}"

for path in \
  "${STAGE_C_V3_TRAIN_MANIFEST}" "${STAGE_C_V3_TRAIN_MANIFEST}.offsets.bin" \
  "${STAGE_C_V3_VALID_MANIFEST}" "${STAGE_C_V3_VALID_MANIFEST}.offsets.bin" \
  "${STAGE_C_V3_STUDENT_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing Stage-C-after-v3 input: ${path}" >&2; exit 1; }
done

if [[ "${MODE}" == "smoke" ]]; then
  output_dir="${STAGE_C_V3_SMOKE_OUTPUT_DIR}"
  tensorboard_dir="${STAGE_C_V3_SMOKE_RUN_DIR}/tensorboard"
  nproc=1
  visible_devices="${STAGE_C_V3_SMOKE_GPU}"
  train_args=(--batch-size 4 --max-steps 2 --eval-interval 1 --eval-batches 1 --calibration-batches 1 --save-interval 1 --log-interval 1 --num-workers 0 --max-audio-seconds 2)
elif [[ "${MODE}" == "throughput" ]]; then
  output_dir="${STAGE_C_V3_THROUGHPUT_OUTPUT_DIR}"
  tensorboard_dir="${STAGE_C_V3_THROUGHPUT_RUN_DIR}/tensorboard"
  nproc=8
  visible_devices="${CUDA_DEVICES}"
  train_args=(--batch-size "${STAGE_C_V3_BATCH_SIZE}" --max-steps 5 --eval-interval 5 --eval-batches 1 --calibration-batches 1 --save-interval 5 --log-interval 1 --num-workers "${STAGE_C_V3_NUM_WORKERS}" --prefetch-factor "${STAGE_C_V3_PREFETCH_FACTOR}")
else
  output_dir="${STAGE_C_V3_OUTPUT_DIR}"
  tensorboard_dir="${STAGE_C_V3_RUN_DIR}/tensorboard"
  nproc=8
  visible_devices="${CUDA_DEVICES}"
  train_args=(--batch-size "${STAGE_C_V3_BATCH_SIZE}" --max-steps "${STAGE_C_V3_MAX_STEPS}" --learning-rate "${STAGE_C_V3_LEARNING_RATE}" --num-workers "${STAGE_C_V3_NUM_WORKERS}" --prefetch-factor "${STAGE_C_V3_PREFETCH_FACTOR}" --eval-interval 250 --eval-batches 4 --calibration-batches 32 --save-interval 250 --log-interval 10)
fi

if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${STAGE_C_V3_RESUME:-0}" != "1" ]]; then
    echo "Refusing to overwrite non-empty output: ${output_dir}" >&2
    exit 1
  fi
fi
mkdir -p "${output_dir}" "${tensorboard_dir}"
resume_args=()
[[ "${STAGE_C_V3_RESUME:-0}" == "1" ]] && resume_args+=(--resume)

monitor="${STAGE_C_V3_LOG_DIR}/gpu_${MODE}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader,nounits --loop=2 > "${monitor}" &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${visible_devices}" torchrun \
  --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}" \
  --master-addr 127.0.0.1 --master-port "${STAGE_C_V3_MASTER_PORT}" \
  -m training.simul_uniss.subsecond_v3.train_stage_c_after_v3 \
  --train-manifest "${STAGE_C_V3_TRAIN_MANIFEST}" \
  --valid-manifest "${STAGE_C_V3_VALID_MANIFEST}" \
  --student-checkpoint "${STAGE_C_V3_STUDENT_CHECKPOINT}" \
  --output-dir "${output_dir}" \
  --tensorboard-dir "${tensorboard_dir}" \
  --device cuda --bf16 \
  "${train_args[@]}" "${resume_args[@]}" \
  2>&1 | tee "${STAGE_C_V3_LOG_DIR}/${MODE}_train.log"
