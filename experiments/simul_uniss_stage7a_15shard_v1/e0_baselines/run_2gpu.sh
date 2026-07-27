#!/usr/bin/env bash
set -euo pipefail

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"

RUN_ID="$([[ "${SMOKE}" == 1 ]] && echo "smoke_$(date -u +%Y%m%dT%H%M%SZ)" || echo full)"
OUTPUT_DIR="${EVAL_ROOT}/e0_baselines_${RUN_ID}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}/stage6/dev" "${OUTPUT_DIR}/logs"
LIMIT="$([[ "${SMOKE}" == 1 ]] && echo 32 || echo 0)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

"${TRAIN_ENV}/bin/python" -m training.simul_uniss.stage7a.fixed_policy \
  --schedules "${DEV_SCHEDULES}" \
  --output "${OUTPUT_DIR}/fixed_wait_k_dev.json" \
  --wait-k 1 2 3 5 --limit-records "${LIMIT}" \
  > "${OUTPUT_DIR}/logs/fixed_wait_k.log" 2>&1

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 2 > "${OUTPUT_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

ARGS=(
  --model "${STAGE6_MODEL}"
  --samples "${DEV_SAMPLES}"
  --schedules "${DEV_SCHEDULES}"
  --output-dir "${OUTPUT_DIR}/stage6/dev"
  --split dev
  --dtype bf16
  --attention-implementation flash_attention_2
  --max-batch-tokens "${EVAL_MAX_BATCH_TOKENS_PER_GPU}"
  --max-batch-size "${EVAL_MAX_BATCH_SIZE_PER_GPU}"
  --logit-event-batch 256
  --warmup-batches 2
  --warmup-batch-size 16
  --progress-interval 10
)
if [[ "${LIMIT}" != 0 ]]; then ARGS+=(--limit-records "${LIMIT}"); fi

CUDA_VISIBLE_DEVICES="${E0_GPUS}" "${TRAIN_ENV}/bin/torchrun" \
  --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
  --master-port "${E0_MASTER_PORT}" --nproc-per-node 2 \
  -m evaluation.simultaneous_streaming.stage3_action_eval "${ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/stage6_action_eval.log"

cleanup
trap - EXIT
echo "OUTPUT_DIR=${OUTPUT_DIR}"
