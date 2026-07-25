#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/experiment.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

assemble_cmd=(python -m training.simul_uniss.full_data_pipeline assemble
  --prepared-parts "${PREPARED_PARTS_DIR}"
  --packed-parts "${PACKED_PARTS_DIR}"
  --shard-start "${SHARD_START}" --shard-count "${SHARD_COUNT}"
  --schedules-output "${SCHEDULES_JSONL}"
  --interleaved-output "${PACKED_TRAIN}"
  --action-output "${ACTION_PACKED_TRAIN}"
  --manifest-output "${FULL_DATA_MANIFEST}"
  --marker-output "${DATA_ASSEMBLY_MARKER}")
eval_cmd=(python -m training.simul_uniss.stage0_eval
  --input "${SCHEDULES_JSONL}" --output "${STAGE0_METRICS}"
  --tensorboard-dir "${TENSORBOARD_DIR}/stage00_data_metrics"
  --limit-records "${STAGE0_EVAL_RECORDS}")
schedule_cmd=(python -m training.simul_uniss.full_data_pipeline schedule
  --assembly-marker "${DATA_ASSEMBLY_MARKER}"
  --output "${TRAINING_SCHEDULE_FILE}"
  --global-batch-size "${SIMUL_GLOBAL_BATCH_SIZE}"
  --stage3-epochs "${FULL_STAGE3_EPOCHS}"
  --stage4-epochs "${FULL_STAGE4_EPOCHS}"
  --stage6-epochs "${FULL_STAGE6_EPOCHS}"
  --warmup-fraction "${FULL_WARMUP_FRACTION}")
finalize_cmd=(python -m training.simul_uniss.full_data_pipeline finalize
  --assembly-marker "${DATA_ASSEMBLY_MARKER}"
  --schedule-file "${TRAINING_SCHEDULE_FILE}"
  --stage0-metrics "${STAGE0_METRICS}"
  --output "${FULL_DATA_READY_MARKER}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${assemble_cmd[@]}"; printf '\n'
  printf '%q ' "${eval_cmd[@]}"; printf '\n'
  printf '%q ' "${schedule_cmd[@]}"; printf '\n'
  printf '%q ' "${finalize_cmd[@]}"; printf '\n'
  exit 0
fi
mkdir -p "${PROCESSED_DIR}" "${PACKED_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"
{
  "${assemble_cmd[@]}"
  "${eval_cmd[@]}"
  "${schedule_cmd[@]}"
  "${finalize_cmd[@]}"
} 2>&1 | tee -a "${LOG_DIR}/data_preparation/assemble_full198.log"
