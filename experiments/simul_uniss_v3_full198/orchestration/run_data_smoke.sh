#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
SMOKE_ROOT="${DATA_SMOKE_ROOT:-${RUN_DIR}/data_smoke_v3}"
if [[ -f "${SMOKE_ROOT}/DATA_SMOKE_COMPLETE" ]]; then
  [[ -f "${SMOKE_ROOT}/packed/FULL_DATA_READY.json" ]] || { echo "Incomplete existing smoke: ${SMOKE_ROOT}" >&2; exit 1; }
  echo "Full-data preparation smoke already completed: ${SMOKE_ROOT}"
  exit 0
fi
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "Refusing partial smoke output: ${SMOKE_ROOT}" >&2; exit 1; }

export SHARD_START=0 SHARD_COUNT=2 PREPARE_WORKERS=2 PACK_WORKERS=2
export PROCESSED_DIR="${SMOKE_ROOT}/processed" PACKED_DIR="${SMOKE_ROOT}/packed"
export PREPARED_PARTS_DIR="${PROCESSED_DIR}/parts" PACKED_PARTS_DIR="${PACKED_DIR}/parts"
export SCHEDULES_JSONL="${PROCESSED_DIR}/schedules.jsonl"
export PACKED_TRAIN="${PACKED_DIR}/packed_interleaved_train.jsonl"
export ACTION_PACKED_TRAIN="${PACKED_DIR}/packed_action_train.jsonl"
export FULL_DATA_MANIFEST="${PROCESSED_DIR}/full_data_manifest.json"
export DATA_ASSEMBLY_MARKER="${PACKED_DIR}/DATA_ASSEMBLY_COMPLETE.json"
export FULL_DATA_READY_MARKER="${PACKED_DIR}/FULL_DATA_READY.json"
export TRAINING_SCHEDULE_FILE="${PACKED_DIR}/training_schedule.env"
export STAGE0_METRICS="${SMOKE_ROOT}/stage00_metrics.json" STAGE0_EVAL_RECORDS=8
export TENSORBOARD_DIR="${SMOKE_ROOT}/tensorboard" LOG_DIR="${SMOKE_ROOT}/logs"

for index in 0 1; do
  "${EXPERIMENT_DIR}/data_preparation/prepare_one_shard.sh" \
    --config "${EXPERIMENT_DIR}/experiment.env" --index "${index}" --limit-records 4
done
for index in 0 1; do
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" \
    --config "${EXPERIMENT_DIR}/experiment.env" --index "${index}"
done
"${EXPERIMENT_DIR}/data_preparation/assemble_full198.sh" --config "${EXPERIMENT_DIR}/experiment.env"
python - "${DATA_ASSEMBLY_MARKER}" "${FULL_DATA_READY_MARKER}" <<'PY'
import json
import sys
from pathlib import Path

assembly = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert assembly["shard_count"] == 2
assert assembly["source_records"] == 8
assert assembly["packed_interleaved_records"] > 0
assert assembly["packed_action_records"] > 0
assert Path(sys.argv[2]).is_file()
PY
printf 'completed_at=%s\nroot=%s\n' "$(date -u +%FT%TZ)" "${SMOKE_ROOT}" > "${SMOKE_ROOT}/DATA_SMOKE_COMPLETE"
echo "Full-data preparation smoke completed: ${SMOKE_ROOT}"
