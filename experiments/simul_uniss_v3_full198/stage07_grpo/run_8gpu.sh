#!/usr/bin/env bash
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
cmd=(torchrun --nproc_per_node "${SIMUL_NPROC_PER_NODE}"
  --master_addr "${SIMUL_MASTER_ADDR}" --master_port "${STAGE7_MASTER_PORT}"
  -m training.simul_uniss.policy_grpo --schedules "${SCHEDULES_JSONL}"
  --output-dir "${STAGE7_OUTPUT_DIR}" --tensorboard-dir "${STAGE7_TENSORBOARD_DIR}"
  --device cuda --batch-size "${STAGE7_BATCH_SIZE}" --sft-steps "${STAGE7_SFT_STEPS}"
  --grpo-steps "${STAGE7_GRPO_STEPS}" --group-size "${STAGE7_GROUP_SIZE}"
  --shuffle-buffer-size "${SIMUL_ITERABLE_SHUFFLE_BUFFER_SIZE}" --seed "${SIMUL_DATA_SEED}")
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "${FULL_DATA_READY_MARKER}" && -f "${SCHEDULES_JSONL}" ]] || { echo "Full schedules are not ready" >&2; exit 1; }
[[ ! -e "${STAGE7_OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${STAGE7_OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}" "${STAGE7_TENSORBOARD_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage07_grpo_policy_bootstrap.log"
