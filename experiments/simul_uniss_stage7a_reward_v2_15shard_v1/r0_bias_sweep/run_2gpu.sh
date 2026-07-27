#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
SMOKE="${1:-}"
RUN_SUFFIX=full_dev
BIASES="${R0_BIASES}"
LIMIT_ARGS=()
if [[ "${SMOKE}" == "--smoke" ]]; then
  RUN_SUFFIX="smoke_$(date -u +%Y%m%dT%H%M%SZ)"
  BIASES="0.00 0.20"
  LIMIT_ARGS=(--limit-records 64)
elif [[ -n "${SMOKE}" ]]; then
  echo "Usage: $0 [--smoke]" >&2
  exit 2
fi
RUN_DIR="${EVAL_ROOT}/r0_e3_v1_bias_sweep_${RUN_SUFFIX}"
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to overwrite ${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}/logs"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 2 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

index=0
for bias in ${BIASES}; do
  label="$(printf '%s' "${bias}" | tr '.-' 'pm')"
  output="${RUN_DIR}/bias_${label}.json"
  CUDA_VISIBLE_DEVICES="${R0_GPUS}" "${TRAIN_ENV}/bin/torchrun" \
    --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
    --master-port "$((R0_MASTER_PORT + index))" --nproc-per-node 2 \
    -m training.simul_uniss.stage7a.evaluate \
    --checkpoint "${E3_V1_CHECKPOINT}" --samples "${DEV_SAMPLES}" \
    --output "${output}" --device cuda --dtype bf16 \
    --attention-implementation flash_attention_2 \
    --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
    --max-batch-tokens "${EVAL_MAX_BATCH_TOKENS_PER_GPU}" \
    --max-batch-size "${EVAL_MAX_BATCH_SIZE_PER_GPU}" \
    --write-logit-bias "${bias}" "${LIMIT_ARGS[@]}" \
    > "${RUN_DIR}/logs/bias_${label}.log" 2>&1
  index=$((index + 1))
done

"${TRAIN_ENV}/bin/python" "${ROOT}/r0_bias_sweep/report.py" \
  --input-dir "${RUN_DIR}" --output-json "${RUN_DIR}/bias_sweep.json" \
  --report "${RUN_DIR}/bias_sweep_report.md" \
  --selected-bias "${RUN_DIR}/selected_bias.txt"
touch "${RUN_DIR}/COMPLETE"
cleanup; trap - EXIT
if [[ "${SMOKE}" != "--smoke" ]]; then
  SELECTED_BIAS="$(tr -d '[:space:]' < "${RUN_DIR}/selected_bias.txt")"
  "${ROOT}/evaluation/run_dev_2gpu.sh" \
    r0_e3_v1_bias "${E3_V1_MODEL}" "${R0_GPUS}" "${SELECTED_BIAS}"
fi
echo "RUN_DIR=${RUN_DIR}"
