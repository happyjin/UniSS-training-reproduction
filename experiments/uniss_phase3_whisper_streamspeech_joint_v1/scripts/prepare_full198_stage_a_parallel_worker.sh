#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

WORKER_ID="${1:?worker id is required}"
TOTAL_WORKERS="${2:?total worker count is required}"
GPU_ID="${3:?physical GPU id is required}"
if (( WORKER_ID < 0 || TOTAL_WORKERS <= 0 || WORKER_ID >= TOTAL_WORKERS )); then
  echo "invalid worker geometry: ${WORKER_ID}/${TOTAL_WORKERS}" >&2
  exit 1
fi

PARQUET_BATCH_SIZE="${PARQUET_BATCH_SIZE:-128}"
IO_WORKERS="${IO_WORKERS:-4}"
for ((shard=WORKER_ID; shard<198; shard+=TOTAL_WORKERS)); do
  name="$(printf 'train-%05d' "${shard}")"
  input="${UNIST_RAW_ROOT}/${name}.parquet"
  output="${FORMAL_STAGE_A_ROOT}/parts/${name}"
  require_file "${input}"
  if [[ -f "${output}/STAGE_A_SOURCE_PART_COMPLETE.json" ]]; then
    echo "already complete: ${name}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" \
    -m training.simul_uniss.subsecond_v1.stage_a prepare-part \
    --input "${input}" \
    --output-dir "${output}" \
    --bicodec-checkpoint "${BICODEC_CHECKPOINT}" \
    --device cuda:0 \
    --side source \
    --batch-size "${PARQUET_BATCH_SIZE}" \
    --io-workers "${IO_WORKERS}" \
    --progress-interval 10000 \
    --skip-sha256
done
