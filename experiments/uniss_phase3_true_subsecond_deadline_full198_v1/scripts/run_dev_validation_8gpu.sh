#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config.env"

DEV_LOG_ROOT="${REPO_ROOT}/logs/${EXPERIMENT_NAME}/dev_trajectory_cache"
DEV_SUMMARY="${DEV_CACHE_ROOT}/trajectory_summary.json"
DEV_ASSEMBLY_MARKER="${DEV_PACKED_ROOT}/ASSEMBLY_COMPLETE.json"
mkdir -p \
  "${DEV_CACHE_ROOT}" "${DEV_PACKED_ROOT}/parts" "${DEV_LOG_ROOT}" \
  /opt/dlami/nvme/jasonleeeli/tmp /opt/dlami/nvme/jasonleeeli/hf_cache

bash "${SCRIPT_DIR}/prepare_dev_validation_cpu.sh"
if [[ ! -f "${DEV_ASSEMBLY_MARKER}" ]]; then
  monitor_pid=""
  cleanup() {
    if [[ -n "${monitor_pid}" ]]; then
      kill "${monitor_pid}" 2>/dev/null || true
      wait "${monitor_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
    --format=csv -l 5 > "${DEV_LOG_ROOT}/gpu_telemetry.csv" &
  monitor_pid=$!

  pids=()
  for rank in $(seq 0 7); do
    rank_log="${DEV_LOG_ROOT}/rank$(printf '%02d' "${rank}").log"
    (
      export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
      export HF_HOME=/opt/dlami/nvme/jasonleeeli/hf_cache
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
      "${PYTHON}" -m \
        experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache \
        --raw-unist-dir "${RAW_UNIST_DIR}" \
        --source-template dev-00000.parquet \
        --index-root "${DEV_INDEX_ROOT}" \
        --index-template 'part-{shard:03d}.{lang}.npy' \
        --output-root "${DEV_CACHE_ROOT}" \
        --phase3-model "${PHASE3_MODEL}" \
        --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
        --bicodec-checkpoint "${REPO_ROOT}/pretrained_models/UniSS/bicodec/BiCodec" \
        --rank "${rank}" --world-size 8 --shard-count "${DEV_SHARD_COUNT}" \
        --batch-size 64 --topk 32 --temperature 1.5 \
        --confidence-threshold 0.70 --progress-interval 512
    ) > "${rank_log}" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  (( status == 0 )) || {
    echo "one or more dev cache ranks failed; inspect ${DEV_LOG_ROOT}" >&2
    exit "${status}"
  }
  cleanup
  monitor_pid=""

  "${PYTHON}" -m \
    experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.validate_trajectory_cache \
    --root "${DEV_CACHE_ROOT}" --output "${DEV_SUMMARY}" \
    --shard-count "${DEV_SHARD_COUNT}"

  "${PYTHON}" -m \
    experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_completed_shards \
    --cache-root "${DEV_CACHE_ROOT}" --raw-root "${RAW_UNIST_DIR}" \
    --raw-template dev-00000.parquet --parts-root "${DEV_PACKED_ROOT}/parts" \
    --shard-count "${DEV_SHARD_COUNT}" --seq-length "${SEQ_LENGTH}" \
    --workers "${DEV_SHARD_COUNT}" --poll-seconds 1

  "${PYTHON}" -m \
    experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs \
    --parts-root "${DEV_PACKED_ROOT}/parts" \
    --output "${DEV_TRAJECTORY_PACKED}" \
    --offsets "${DEV_TRAJECTORY_OFFSETS}" \
    --marker "${DEV_ASSEMBLY_MARKER}" \
    --shard-count "${DEV_SHARD_COUNT}" --seq-length "${SEQ_LENGTH}"
fi

for required in \
  "${DEV_SUMMARY}" "${DEV_ASSEMBLY_MARKER}" \
  "${DEV_TRAJECTORY_PACKED}" "${DEV_TRAJECTORY_OFFSETS}"; do
  [[ -f "${required}" ]] || { echo "Incomplete dev validation artifact: ${required}" >&2; exit 1; }
done
echo "full canonical dev trajectory validation artifact is ready: ${DEV_TRAJECTORY_PACKED}"
