#!/usr/bin/env bash
set -euo pipefail

SMOKE=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

cmd=(python -m training.simul_uniss.nar_semantic
  --schedules "${SCHEDULES_JSONL}"
  --output-dir "${STAGE8_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE8_TENSORBOARD_DIR}"
  --device "${STAGE8_DEVICE}"
  --batch-size "${STAGE8_BATCH_SIZE}"
  --max-steps "${STAGE8_MAX_STEPS}"
  --shuffle-buffer-size "${SIMUL_ITERABLE_SHUFFLE_BUFFER_SIZE}"
  --seed "${SIMUL_DATA_SEED}"
)
if [[ "${SMOKE}" == "1" ]]; then
  cmd=(python -m training.simul_uniss.nar_semantic
    --schedules "${SCHEDULES_JSONL}"
    --output-dir "${STAGE8_OUTPUT_DIR}/smoke"
    --tensorboard-dir "${STAGE8_TENSORBOARD_DIR}"
    --device cpu
    --batch-size 1
    --max-steps 2
    --hidden-size 32
    --num-layers 1
    --num-heads 4
    --max-text-tokens 16
    --max-semantic-tokens 32
    --shuffle-buffer-size 32
    --seed "${SIMUL_DATA_SEED}"
    --log-interval 1
  )
fi
mkdir -p "${STAGE8_OUTPUT_DIR}" "${STAGE8_TENSORBOARD_DIR}" "${LOG_DIR}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage8_nar.log"
