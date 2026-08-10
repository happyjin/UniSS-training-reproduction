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

nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
