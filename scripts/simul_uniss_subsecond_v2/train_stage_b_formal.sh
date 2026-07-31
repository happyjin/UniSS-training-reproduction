#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_formal_15shard_v1.env}"
MODE="${1:-formal}"
[[ "${MODE}" == "formal" || "${MODE}" == "smoke" ]] || { echo "mode must be formal or smoke" >&2; exit 2; }
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

if [[ "${MODE}" == "smoke" ]]; then
  manifest="${STAGE_B_SMOKE_MANIFEST}"
  valid_manifest="${STAGE_B_SMOKE_VALID_MANIFEST}"
  output_dir="${STAGE_B_SMOKE_ROOT}"
  tensorboard_dir="${STAGE_B_SMOKE_RUN_ROOT}/tensorboard"
  nproc=1
  train_args=(--batch-size 2 --max-steps 2 --learning-rate 1e-3 --hidden-size 128 --num-layers 2 --num-heads 4 --ffn-dim 512 --max-audio-seconds 2 --validation-records 8 --eval-interval 1 --eval-batches 1 --save-interval 1 --log-interval 1 --num-workers 0)
  validate_args=(--samples 8 --smoke)
else
  manifest="${STAGE_B_TRAIN_MANIFEST}"
  valid_manifest="${STAGE_B_VALID_MANIFEST}"
  output_dir="${STAGE_B_ROOT}"
  tensorboard_dir="${STAGE_B_RUN_ROOT}/tensorboard"
  nproc=8
  train_args=(--batch-size "${STAGE_B_BATCH_SIZE}" --max-steps "${STAGE_B_MAX_STEPS}" --learning-rate "${STAGE_B_LEARNING_RATE}" --hidden-size "${STAGE_B_HIDDEN_SIZE}" --num-layers "${STAGE_B_NUM_LAYERS}" --num-heads "${STAGE_B_NUM_HEADS}" --ffn-dim "${STAGE_B_FFN_DIM}" --max-audio-seconds "${STAGE_B_MAX_AUDIO_SECONDS}" --validation-records "${STAGE_B_VALIDATION_RECORDS}" --eval-interval 500 --eval-batches 16 --save-interval 500 --log-interval 10 --num-workers "${STAGE_B_NUM_WORKERS}")
  validate_args=(--samples "${STAGE_B_VALIDATION_SAMPLES}" --minimum-agreement "${STAGE_B_MINIMUM_AGREEMENT}" --maximum-rtf "${STAGE_B_MAXIMUM_RTF}")
fi

for path in "${manifest}" "${manifest}.offsets.bin" "${valid_manifest}" "${valid_manifest}.offsets.bin" "${POLICY_TOKENIZER_MODEL}"; do
  [[ -f "${path}" ]] || { echo "Missing required Stage-B input: ${path}" >&2; exit 1; }
done
mkdir -p "${output_dir}" "${tensorboard_dir}" "${STAGE_B_LOG_ROOT}"
monitor_log="${STAGE_B_LOG_ROOT}/${MODE}_gpu_monitor.csv"
training_log="${STAGE_B_LOG_ROOT}/${MODE}_train.log"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits --loop=2 > "${monitor_log}" &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT

resume_args=()
[[ "${STAGE_B_RESUME:-0}" == "1" ]] && resume_args+=(--resume)
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
  --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}" \
  --master-addr 127.0.0.1 --master-port "${STAGE_B_MASTER_PORT}" \
  -m training.simul_uniss.subsecond_v1.train_stage_b \
  --manifest "${manifest}" \
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}" \
  --output-dir "${output_dir}" \
  --tensorboard-dir "${tensorboard_dir}" \
  --teacher-glm-field "${STAGE_B_TEACHER_FIELD}" \
  --teacher-glm-end-field "${STAGE_B_TEACHER_END_FIELD}" \
  --device cuda --bf16 "${train_args[@]}" "${resume_args[@]}" \
  2>&1 | tee -a "${training_log}"

checkpoint="${output_dir}/best.pt"
[[ -f "${checkpoint}" ]] || checkpoint="${output_dir}/last.pt"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" python -m training.simul_uniss.subsecond_v1.validate_stage_b \
  --checkpoint "${checkpoint}" \
  --manifest "${valid_manifest}" \
  --device cuda:0 \
  --reference-field "${STAGE_B_TEACHER_FIELD}" \
  --compatibility-reference-field "${STAGE_B_COMPATIBILITY_FIELD}" \
  --output "${output_dir}/stage_b_validation.json" \
  --mark-complete "${validate_args[@]}" \
  2>&1 | tee -a "${STAGE_B_LOG_ROOT}/${MODE}_validate.log"

