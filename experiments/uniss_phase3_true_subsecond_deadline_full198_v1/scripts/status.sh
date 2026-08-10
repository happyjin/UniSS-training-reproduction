#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

SESSION="${CACHE_TMUX_SESSION:-uniss_true_subsecond_cache_full198}"
LOG_ROOT="${REPO_ROOT}/logs/${EXPERIMENT_NAME}/trajectory_cache"
complete=0
if [[ -d "${CACHE_ROOT}" ]]; then
  complete="$(find "${CACHE_ROOT}" -mindepth 2 -maxdepth 2 -name PART_COMPLETE.json | wc -l)"
fi

echo "tmux_session=${SESSION}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux_running=yes"
else
  echo "tmux_running=no"
fi
echo "cache_parts=${complete}/198"

for rank in $(seq 0 7); do
  log="${LOG_ROOT}/rank$(printf '%02d' "${rank}").log"
  if [[ -f "${log}" ]]; then
    printf 'rank%02d: ' "${rank}"
    tail -n 1 "${log}"
  fi
done

PACK_SESSION="${PACK_TMUX_SESSION:-uniss_true_subsecond_pack_full198}"
if tmux has-session -t "${PACK_SESSION}" 2>/dev/null; then
  echo "pack_tmux_running=yes"
else
  echo "pack_tmux_running=no"
fi
pack_parts=0
if [[ -d "${PACKED_ROOT}/parts" ]]; then
  pack_parts="$(find "${PACKED_ROOT}/parts" -mindepth 2 -maxdepth 2 -name PACK_COMPLETE.json | wc -l)"
fi
echo "pack_parts=${pack_parts}/198"
PACK_LOG="${PACK_LOG:-${REPO_ROOT}/logs/uniss_true_subsecond_pack_full198.log}"
if [[ -f "${PACK_LOG}" ]]; then
  tail -n 3 "${PACK_LOG}"
fi
dev_cache_parts=0
dev_pack_parts=0
if [[ -d "${DEV_CACHE_ROOT}" ]]; then
  dev_cache_parts="$(find "${DEV_CACHE_ROOT}" -mindepth 2 -maxdepth 2 -name PART_COMPLETE.json | wc -l)"
fi
if [[ -d "${DEV_PACKED_ROOT}/parts" ]]; then
  dev_pack_parts="$(find "${DEV_PACKED_ROOT}/parts" -mindepth 2 -maxdepth 2 -name PACK_COMPLETE.json | wc -l)"
fi
echo "dev_cache_parts=${dev_cache_parts}/${DEV_SHARD_COUNT}"
echo "dev_pack_parts=${dev_pack_parts}/${DEV_SHARD_COUNT}"
if [[ -f "${DEV_PACKED_ROOT}/ASSEMBLY_COMPLETE.json" ]]; then
  echo "dev_validation_assembly=complete"
else
  echo "dev_validation_assembly=pending"
fi
PIPELINE_SESSION="${PIPELINE_TMUX_SESSION:-uniss_true_subsecond_full198_pipeline}"
if tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null; then
  echo "pipeline_tmux_running=yes"
else
  echo "pipeline_tmux_running=no"
fi
if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "training_iteration=$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
fi

nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
