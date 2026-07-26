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

cmd=(python -m training.simul_uniss.full_data_pipeline schedule
  --assembly-marker "${DATA_ASSEMBLY_MARKER}"
  --output "${TRAINING_SCHEDULE_FILE}"
  --global-batch-size "${SIMUL_GLOBAL_BATCH_SIZE}"
  --stage3-epochs "${FULL_STAGE3_EPOCHS}"
  --stage4-epochs "${FULL_STAGE4_EPOCHS}"
  --stage6-epochs "${FULL_STAGE6_EPOCHS}"
  --warmup-fraction "${FULL_WARMUP_FRACTION}")
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "${FULL_DATA_READY_MARKER}" && -f "${DATA_ASSEMBLY_MARKER}" ]] || {
  echo "Source full198 data is not ready" >&2; exit 1;
}
[[ ! -e "${TRAINING_SCHEDULE_FILE}" ]] || {
  echo "Refusing to overwrite ${TRAINING_SCHEDULE_FILE}" >&2; exit 1;
}
mkdir -p "$(dirname "${TRAINING_SCHEDULE_FILE}")"
"${cmd[@]}"
echo "Generated isolated GBS128 schedule: ${TRAINING_SCHEDULE_FILE}"
