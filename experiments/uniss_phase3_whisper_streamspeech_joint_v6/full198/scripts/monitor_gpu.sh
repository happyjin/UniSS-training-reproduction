#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:?usage: monitor_gpu.sh OUTPUT.csv}"
INTERVAL="${GPU_MONITOR_INTERVAL:-10}"
mkdir -p "$(dirname "${OUTPUT}")"
if [[ ! -e "${OUTPUT}" ]]; then
  echo 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w,power_limit_w' >"${OUTPUT}"
fi
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits | sed "s/^/${timestamp},/" >>"${OUTPUT}"
  sleep "${INTERVAL}"
done
