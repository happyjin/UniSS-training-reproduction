#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
MODE="formal"
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --smoke) MODE="smoke"; shift ;;
    --formal) MODE="formal"; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_ab.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${LOG_ROOT}/stage_b"
if [[ "${MODE}" == "smoke" ]]; then
  manifest="${STAGE_A_SMOKE_ROOT}/stage_a_source_manifest.jsonl"
  output_dir="${STAGE_B_SMOKE_ROOT}"
  tensorboard_dir="${STAGE_B_SMOKE_RUN_ROOT}/tensorboard"
  train_args=(
    --batch-size 2
    --max-steps 10
    --learning-rate 1e-3
    --hidden-size 128
    --num-layers 2
    --num-heads 4
    --ffn-dim 512
    --max-audio-seconds 2
    --validation-records 8
    --eval-interval 5
    --eval-batches 2
    --save-interval 5
    --log-interval 1
    --num-workers 0
  )
else
  manifest="${STAGE_A_ROOT}/stage_a_source_manifest.jsonl"
  output_dir="${STAGE_B_ROOT}"
  tensorboard_dir="${STAGE_B_RUN_ROOT}/tensorboard"
  train_args=(
    --batch-size "${STAGE_B_BATCH_SIZE}"
    --max-steps "${STAGE_B_MAX_STEPS}"
    --learning-rate "${STAGE_B_LEARNING_RATE}"
    --hidden-size "${STAGE_B_HIDDEN_SIZE}"
    --num-layers "${STAGE_B_NUM_LAYERS}"
    --num-heads "${STAGE_B_NUM_HEADS}"
    --ffn-dim "${STAGE_B_FFN_DIM}"
    --max-audio-seconds "${STAGE_B_MAX_AUDIO_SECONDS}"
    --validation-records "${STAGE_B_VALIDATION_RECORDS}"
    --eval-interval 500
    --eval-batches 16
    --save-interval 500
    --log-interval 10
    --num-workers "${STAGE_B_NUM_WORKERS}"
  )
fi

[[ -f "${manifest}" ]] || { echo "Missing Stage A manifest: ${manifest}" >&2; exit 1; }
mkdir -p "${output_dir}" "${tensorboard_dir}"
monitor_log="${LOG_ROOT}/stage_b/${MODE}_gpu_monitor.csv"
training_log="${LOG_ROOT}/stage_b/${MODE}_train.log"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader,nounits --loop=2 > "${monitor_log}" &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT

cmd=(torchrun
  --nnodes 1
  --node-rank 0
  --nproc-per-node 8
  --master-addr 127.0.0.1
  --master-port "${STAGE_B_MASTER_PORT}"
  -m training.simul_uniss.subsecond_v1.train_stage_b
  --manifest "${manifest}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --output-dir "${output_dir}"
  --tensorboard-dir "${tensorboard_dir}"
  --device cuda
  --bf16
  "${train_args[@]}"
)
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}" 2>&1 | tee -a "${training_log}"
cleanup
trap - EXIT

checkpoint="${output_dir}/best.pt"
if [[ ! -f "${checkpoint}" ]]; then
  checkpoint="${output_dir}/last.pt"
fi
validation_args=()
if [[ "${MODE}" == "smoke" ]]; then
  validation_args+=(--smoke)
fi
CUDA_VISIBLE_DEVICES=2 python -m training.simul_uniss.subsecond_v1.validate_stage_b \
  --checkpoint "${checkpoint}" \
  --manifest "${manifest}" \
  --device cuda:0 \
  --samples 16 \
  --output "${output_dir}/stage_b_validation.json" \
  --mark-complete \
  "${validation_args[@]}" \
  2>&1 | tee -a "${LOG_ROOT}/stage_b/${MODE}_validate.log"
