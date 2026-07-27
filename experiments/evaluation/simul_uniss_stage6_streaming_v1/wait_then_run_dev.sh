#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 2
fi
RUN_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
LOG_DIR="${REPO_ROOT}/logs/evaluation/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/${RUN_ID}.log") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for Stage4 test COMPLETE: ${STAGE4_TEST_RUN}/COMPLETE"
while [[ ! -f "${STAGE4_TEST_RUN}/COMPLETE" ]]; do
  sleep "${GPU_IDLE_POLL_SECONDS}"
done

idle_polls=0
while (( idle_polls < GPU_IDLE_POLLS )); do
  if nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
      awk -F',' -v max_mem="${GPU_IDLE_MEMORY_MIB}" -v max_util="${GPU_IDLE_UTILIZATION}" '
        {gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)}
        $1 >= 4 && $1 <= 7 {seen++; if ($2 >= max_mem || $3 >= max_util) busy=1}
        END {exit !(seen == 4 && !busy)}'; then
    idle_polls=$((idle_polls + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU 4-7 idle poll ${idle_polls}/${GPU_IDLE_POLLS}"
  else
    idle_polls=0
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU 4-7 still busy; waiting"
  fi
  if (( idle_polls < GPU_IDLE_POLLS )); then sleep "${GPU_IDLE_POLL_SECONDS}"; fi
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching Stage6 dev on GPU 4-7"
"${SCRIPT_DIR}/run_full_split_4gpu.sh" dev "${RUN_ID}"
