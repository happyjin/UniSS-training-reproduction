#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
CACHE_PROGRESS_INTERVAL="${CACHE_PROGRESS_INTERVAL:-1024}"
CACHE_LOG_DIR="${REPO_ROOT}/logs/${EXPERIMENT_NAME}/trajectory_cache"
CACHE_SUMMARY="${CACHE_ROOT}/trajectory_summary.json"
GPU_MONITOR_LOG="${CACHE_LOG_DIR}/gpu_telemetry.csv"

for required in \
  "${PYTHON}" \
  "${INDEX_ROOT}/index.json" \
  "${PHASE3_MODEL}/model.safetensors" \
  "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer/model.safetensors" \
  "${REPO_ROOT}/pretrained_models/UniSS/bicodec/BiCodec/model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required cache input: ${required}" >&2
    exit 2
  fi
done

mkdir -p \
  "${CACHE_ROOT}" \
  "${CACHE_LOG_DIR}" \
  /opt/dlami/nvme/jasonleeeli/tmp \
  /opt/dlami/nvme/jasonleeeli/hf_cache

if [[ -f "${CACHE_SUMMARY}" ]]; then
  "${PYTHON}" -m \
    experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.validate_trajectory_cache \
    --root "${CACHE_ROOT}" \
    --output "${CACHE_SUMMARY}" \
    --shard-count 198
  exit 0
fi

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv -l 10 > "${GPU_MONITOR_LOG}" &
MONITOR_PID=$!
PIDS=()

cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 7); do
  rank_log="${CACHE_LOG_DIR}/rank$(printf '%02d' "${rank}").log"
  (
    export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
    export HF_HOME=/opt/dlami/nvme/jasonleeeli/hf_cache
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    "${PYTHON}" -m \
      experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache \
      --raw-unist-dir "${RAW_UNIST_DIR}" \
      --index-root "${INDEX_ROOT}" \
      --output-root "${CACHE_ROOT}" \
      --phase3-model "${PHASE3_MODEL}" \
      --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
      --bicodec-checkpoint "${REPO_ROOT}/pretrained_models/UniSS/bicodec/BiCodec" \
      --rank "${rank}" \
      --world-size 8 \
      --batch-size "${CACHE_BATCH_SIZE}" \
      --topk 32 \
      --temperature 1.5 \
      --confidence-threshold 0.70 \
      --progress-interval "${CACHE_PROGRESS_INTERVAL}"
  ) > "${rank_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more trajectory cache ranks failed; inspect ${CACHE_LOG_DIR}" >&2
  exit "${status}"
fi

"${PYTHON}" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.validate_trajectory_cache \
  --root "${CACHE_ROOT}" \
  --output "${CACHE_SUMMARY}" \
  --shard-count 198
