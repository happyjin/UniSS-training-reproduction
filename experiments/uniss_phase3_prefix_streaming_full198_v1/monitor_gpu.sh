#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env"
mkdir -p "$(dirname "${GPU_LOG}")"
if [[ ! -s "${GPU_LOG}" ]]; then
  echo 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w,power_limit_w' > "${GPU_LOG}"
fi
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits >> "${GPU_LOG}"
  sleep 5
done

