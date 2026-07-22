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

cmd=(python -m training.simul_uniss.policy_grpo
  --schedules "${SCHEDULES_JSONL}"
  --output-dir "${STAGE7_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE7_TENSORBOARD_DIR}"
  --device "${STAGE7_DEVICE}"
  --batch-size "${STAGE7_BATCH_SIZE}"
  --sft-steps "${STAGE7_SFT_STEPS}"
  --grpo-steps "${STAGE7_GRPO_STEPS}"
  --group-size "${STAGE7_GROUP_SIZE}"
)
if [[ "${SMOKE}" == "1" ]]; then
  cmd=(python -m training.simul_uniss.policy_grpo
    --schedules "${SCHEDULES_JSONL}"
    --output-dir "${STAGE7_OUTPUT_DIR}/smoke"
    --tensorboard-dir "${STAGE7_TENSORBOARD_DIR}"
    --device cpu
    --batch-size 4
    --sft-steps 2
    --grpo-steps 2
    --group-size 4
    --log-interval 1
  )
fi
mkdir -p "${STAGE7_OUTPUT_DIR}" "${STAGE7_TENSORBOARD_DIR}" "${LOG_DIR}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage7_grpo.log"
