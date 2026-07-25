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

cmd=(torchrun
  --nproc_per_node "${SIMUL_NPROC_PER_NODE}"
  --master_addr "${SIMUL_MASTER_ADDR}"
  --master_port "${STAGE8_MASTER_PORT}"
  -m training.simul_uniss.nar_semantic
  --schedules "${SCHEDULES_JSONL}"
  --output-dir "${STAGE8_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE8_TENSORBOARD_DIR}"
  --device cuda
  --batch-size "${STAGE8_BATCH_SIZE}"
  --max-steps "${STAGE8_MAX_STEPS}"
  --shuffle-buffer-size "${SIMUL_ITERABLE_SHUFFLE_BUFFER_SIZE}"
  --seed "${SIMUL_DATA_SEED}")
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "${SCHEDULES_JSONL}" ]] || { echo "Missing schedules: ${SCHEDULES_JSONL}" >&2; exit 1; }
[[ ! -e "${STAGE8_OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${STAGE8_OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}" "${STAGE8_TENSORBOARD_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage08_nar_semantic_optional.log"
