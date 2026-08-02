#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_LATENT_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_latent_formal_15shard_v1.env}"
MODE="${1:-formal}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1
[[ "${MODE}" == "formal" || "${MODE}" == "smoke" ]] || {
  echo "mode must be formal or smoke" >&2
  exit 2
}
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${STAGE_B_LATENT_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${STAGE_B_LATENT_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${STAGE_B_LATENT_CPU_THREADS}}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

if [[ "${MODE}" == "smoke" ]]; then
  manifest="${STAGE_B_LATENT_SMOKE_MANIFEST}"
  valid_manifest="${STAGE_B_LATENT_SMOKE_VALID_MANIFEST}"
  output_dir="${STAGE_B_LATENT_SMOKE_ROOT}"
  tensorboard_dir="${STAGE_B_LATENT_SMOKE_RUN_ROOT}/tensorboard"
  nproc=1
  train_args=(
    --batch-size 2 --max-steps 2 --learning-rate 1e-3
    --hidden-size 128 --num-layers 2 --num-heads 4 --ffn-dim 512
    --max-audio-seconds 2 --eval-interval 1 --eval-batches 1
    --save-interval 1 --log-interval 1 --num-workers 0
    --consistency-interval 1 --quantize-chunk-size 128
    --latent-weight "${STAGE_B_LATENT_WEIGHT}"
    --hidden-distill-weight "${STAGE_B_HIDDEN_DISTILL_WEIGHT}"
    --source-weight "${STAGE_B_SOURCE_WEIGHT}"
    --capacity-weight "${STAGE_B_CAPACITY_WEIGHT}"
    --stability-weight "${STAGE_B_STABILITY_WEIGHT}"
    --consistency-weight "${STAGE_B_CONSISTENCY_WEIGHT}"
  )
  validate_args=(
    --samples 2 --latency-samples 1 --smoke
    --minimum-correct-stable-coverage "${STAGE_B_LATENT_MINIMUM_CORRECT_STABLE_COVERAGE}"
  )
else
  [[ -f "${STAGE_A_COMPLETE_MARKER}" ]] || {
    echo "Missing completed formal Stage A: ${STAGE_A_COMPLETE_MARKER}" >&2
    exit 1
  }
  manifest="${STAGE_B_LATENT_TRAIN_MANIFEST}"
  valid_manifest="${STAGE_B_LATENT_VALID_MANIFEST}"
  output_dir="${STAGE_B_LATENT_ROOT}"
  tensorboard_dir="${STAGE_B_LATENT_RUN_ROOT}/tensorboard"
  nproc=8
  train_args=(
    --batch-size "${STAGE_B_LATENT_BATCH_SIZE}"
    --max-steps "${STAGE_B_LATENT_MAX_STEPS}"
    --learning-rate "${STAGE_B_LATENT_LEARNING_RATE}"
    --hidden-size "${STAGE_B_LATENT_HIDDEN_SIZE}"
    --num-layers "${STAGE_B_LATENT_NUM_LAYERS}"
    --num-heads "${STAGE_B_LATENT_NUM_HEADS}"
    --ffn-dim "${STAGE_B_LATENT_FFN_DIM}"
    --max-audio-seconds "${STAGE_B_LATENT_MAX_AUDIO_SECONDS}"
    --eval-interval 500 --eval-batches "${STAGE_B_LATENT_EVAL_BATCHES}"
    --save-interval 500 --log-interval 10
    --num-workers "${STAGE_B_LATENT_NUM_WORKERS}"
    --latent-weight "${STAGE_B_LATENT_WEIGHT}"
    --hidden-distill-weight "${STAGE_B_HIDDEN_DISTILL_WEIGHT}"
    --source-weight "${STAGE_B_SOURCE_WEIGHT}"
    --capacity-weight "${STAGE_B_CAPACITY_WEIGHT}"
    --stability-weight "${STAGE_B_STABILITY_WEIGHT}"
    --consistency-weight "${STAGE_B_CONSISTENCY_WEIGHT}"
    --consistency-interval "${STAGE_B_CONSISTENCY_INTERVAL}"
  )
  validate_args=(
    --samples "${STAGE_B_LATENT_VALIDATION_SAMPLES}"
    --latency-samples "${STAGE_B_LATENT_LATENCY_SAMPLES}"
    --minimum-agreement "${STAGE_B_LATENT_MINIMUM_AGREEMENT}"
    --goal-agreement "${STAGE_B_LATENT_GOAL_AGREEMENT}"
    --maximum-rtf "${STAGE_B_LATENT_MAXIMUM_RTF}"
    --maximum-first-stable-p50-ms "${STAGE_B_LATENT_MAXIMUM_FIRST_P50_MS}"
    --maximum-first-stable-p95-ms "${STAGE_B_LATENT_MAXIMUM_FIRST_P95_MS}"
    --minimum-chunk-invariance "${STAGE_B_LATENT_MINIMUM_CHUNK_INVARIANCE}"
    --minimum-correct-stable-coverage "${STAGE_B_LATENT_MINIMUM_CORRECT_STABLE_COVERAGE}"
  )
fi

for path in \
  "${manifest}" "${manifest}.offsets.bin" "${manifest}.offsets.json" \
  "${valid_manifest}" "${valid_manifest}.offsets.bin" "${valid_manifest}.offsets.json" \
  "${POLICY_TOKENIZER_MODEL}" "${WHISPERVQ_CODEBOOK_MODEL}/model.safetensors"; do
  [[ -f "${path}" ]] || { echo "Missing corrected Stage-B input: ${path}" >&2; exit 1; }
done

resume_args=()
[[ "${STAGE_B_LATENT_RESUME:-0}" == "1" ]] && resume_args+=(--resume)
train_command=(
  torchrun --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}"
  --master-addr 127.0.0.1 --master-port "${STAGE_B_LATENT_MASTER_PORT}"
  -m training.simul_uniss.subsecond_v2.train_stage_b_latent
  --manifest "${manifest}" --valid-manifest "${valid_manifest}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --codebook-model "${WHISPERVQ_CODEBOOK_MODEL}"
  --codebook-key "${WHISPERVQ_CODEBOOK_KEY}"
  --output-dir "${output_dir}" --tensorboard-dir "${tensorboard_dir}"
  --teacher-glm-field "${STAGE_B_LATENT_TEACHER_FIELD}"
  --teacher-glm-end-field "${STAGE_B_LATENT_TEACHER_END_FIELD}"
  --device cuda --bf16 "${train_args[@]}" "${resume_args[@]}"
)

checkpoint="${output_dir}/best.pt"
validate_command=(
  python -m training.simul_uniss.subsecond_v2.validate_stage_b_latent
  --checkpoint "${checkpoint}" --manifest "${valid_manifest}"
  --device cuda:0 --reference-field "${STAGE_B_LATENT_TEACHER_FIELD}"
  --output "${output_dir}/stage_b_latent_validation.json"
  --mark-complete "${validate_args[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICES}"
  printf '%q ' "${train_command[@]}"
  printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICES%%,*}"
  printf '%q ' "${validate_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${output_dir}" "${tensorboard_dir}" "${STAGE_B_LATENT_LOG_ROOT}"
monitor_log="${STAGE_B_LATENT_LOG_ROOT}/${MODE}_gpu_monitor.csv"
training_log="${STAGE_B_LATENT_LOG_ROOT}/${MODE}_train.log"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader,nounits --loop=2 > "${monitor_log}" &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${train_command[@]}" 2>&1 | tee -a "${training_log}"
[[ -f "${checkpoint}" ]] || checkpoint="${output_dir}/last.pt"
validate_command[4]="${checkpoint}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" "${validate_command[@]}" \
  2>&1 | tee -a "${STAGE_B_LATENT_LOG_ROOT}/${MODE}_validate.log"
