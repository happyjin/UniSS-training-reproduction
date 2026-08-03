#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

SOURCE_MANIFEST="${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
WORLD_SIZE="${WORLD_SIZE:-8}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_DEVICES}"
[[ "${#gpu_ids[@]}" -ge "${WORLD_SIZE}" ]] || {
  echo "CUDA_DEVICES has fewer than ${WORLD_SIZE} entries" >&2
  exit 1
}
mkdir -p "${OUTPUT_ROOT}/logs"

pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  CUDA_VISIBLE_DEVICES="${gpu_ids[rank]}" python -m \
    training.simul_uniss.subsecond_v3.prepare_prefix_hidden_sidecar \
    --source-manifest "${SOURCE_MANIFEST}" \
    --selection-manifest "${SELECTION_MANIFEST}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --output-dir "${OUTPUT_ROOT}" \
    --device cuda:0 \
    --rank "${rank}" \
    --world-size "${WORLD_SIZE}" \
    --audio-workers "${V3_AUDIO_WORKERS}" \
    --record-batch-size "${V3_RECORD_BATCH_SIZE}" \
    --records-per-shard "${V3_RECORDS_PER_SHARD}" \
    --chunk-ms 160 --lookahead-ms 80 --codebook-topk 32 \
    > "${OUTPUT_ROOT}/logs/part-${rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" -eq 0 ]] || exit 1

python -m training.simul_uniss.subsecond_v2.assemble_stage_a_v3_sidecar \
  --root "${OUTPUT_ROOT}" --world-size "${WORLD_SIZE}"
