#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

mkdir -p "${CACHE_ROOT}" "${LOG_ROOT}/cache" "${REPORT_ROOT}" \
  "${USER_ROOT}/tmp" "${USER_ROOT}/hf_cache"

ATTEMPT_ID="${CACHE_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TELEMETRY="${LOG_ROOT}/cache/gpu_telemetry.${ATTEMPT_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,power.limit \
  --format=csv -l 5 > "${TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT INT TERM

run_attempt() {
  local batch="$1"
  local status=0
  local pids=()
  ATTEMPT_LOGS=()
  echo "cache attempt batch=${batch} teacher_batch=${TEACHER_REQUEST_BATCH_SIZE}" | tee -a "${LOG_ROOT}/cache/attempts.log"
  for rank in $(seq 0 7); do
    rank_log="${LOG_ROOT}/cache/rank$(printf '%02d' "${rank}").batch${batch}.${ATTEMPT_ID}.log"
    ATTEMPT_LOGS+=("${rank_log}")
    (
      export TMPDIR="${USER_ROOT}/tmp"
      export HF_HOME="${USER_ROOT}/hf_cache"
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
      "${PYTHON}" -m experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.build_cache \
        --raw-unist-dir "${RAW_UNIST_DIR}" \
        --index-root "${INDEX_ROOT}" \
        --output-root "${CACHE_ROOT}" \
        --phase3-model "${PHASE3_MODEL}" \
        --whispervq-model "${WHISPERVQ_MODEL}" \
        --bicodec-checkpoint "${BICODEC_CHECKPOINT}" \
        --rank "${rank}" \
        --world-size 8 \
        --shard-count 15 \
        --dynamic-shard-queue \
        --batch-size "${batch}" \
        --teacher-request-batch-size "${TEACHER_REQUEST_BATCH_SIZE}" \
        --topk 32 \
        --temperature 1.5 \
        --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
        --progress-interval 1024
    ) > "${rank_log}" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

success=0
for batch in ${CACHE_BATCH_CANDIDATES:-160 144 128 96 64}; do
  if run_attempt "${batch}"; then
    success=1
    break
  fi
  if ! rg -q "out of memory|CUDA error: out of memory" "${ATTEMPT_LOGS[@]}"; then
    echo "cache failed for a non-OOM reason at batch ${batch}" >&2
    exit 1
  fi
  echo "OOM at batch ${batch}; retrying unfinished shards with smaller batch" | tee -a "${LOG_ROOT}/cache/attempts.log"
done
[[ "${success}" == 1 ]] || { echo "all cache batch candidates failed" >&2; exit 1; }

"${PYTHON}" -m experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.audit \
  --root "${CACHE_ROOT}" \
  --output "${REPORT_ROOT}/data_audit_v2.json" \
  --shard-count 15
