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
assemble=(python -m training.simul_uniss.repack_interleaved assemble
  --parts-root "${PACKED_PARTS_DIR}" --shard-start "${SHARD_START}" --shard-count "${SHARD_COUNT}"
  --seq-length "${SEQ_LENGTH}" --output "${PACKED_TRAIN}" --marker "${DATA_ASSEMBLY_MARKER}")
validation=(python -m training.simul_uniss.repack_interleaved pack
  --input "${VALID_SOURCE_SAMPLES}" --output "${VALID_PACKED_INTERLEAVED}"
  --marker "${VALID_INTERLEAVED_MARKER}" --seq-length "${SEQ_LENGTH}")
schedule=(python -m training.simul_uniss.repack_interleaved schedule
  --assembly-marker "${DATA_ASSEMBLY_MARKER}" --output "${TRAINING_SCHEDULE_FILE}"
  --global-batch-size "${SIMUL_GLOBAL_BATCH_SIZE}" --stage4-epochs "${FULL_STAGE4_EPOCHS}"
  --stage6-epochs "${FULL_STAGE6_EPOCHS}" --warmup-fraction "${FULL_WARMUP_FRACTION}")
finalize=(python -m training.simul_uniss.repack_interleaved finalize
  --assembly-marker "${DATA_ASSEMBLY_MARKER}" --validation-marker "${VALID_INTERLEAVED_MARKER}"
  --schedule "${TRAINING_SCHEDULE_FILE}" --output "${FULL_DATA_READY_MARKER}")
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${assemble[@]}"; printf '\n'
  printf '%q ' "${validation[@]}"; printf '\n'
  printf '%q ' "${schedule[@]}"; printf '\n'
  printf '%q ' "${finalize[@]}"; printf '\n'
  exit 0
fi
mkdir -p "${PACKED_DIR}" "${VALID_PACKED_DIR}" "${LOG_DIR}/data_preparation"
{
  "${assemble[@]}"
  "${validation[@]}"
  "${schedule[@]}"
  "${finalize[@]}"
} 2>&1 | tee -a "${LOG_DIR}/data_preparation/assemble.log"
