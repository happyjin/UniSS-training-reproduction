#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

SMOKE_ROOT="${DATA_ROOT}/smoke/cache_1gpu"
if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "refusing to overwrite existing smoke output: ${SMOKE_ROOT}" >&2
  exit 2
fi
mkdir -p "${SMOKE_ROOT}" "${LOG_ROOT}/smoke"
export SMOKE_ROOT

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m \
  experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.build_cache \
  --raw-unist-dir "${RAW_UNIST_DIR}" \
  --index-root "${INDEX_ROOT}" \
  --output-root "${SMOKE_ROOT}" \
  --phase3-model "${PHASE3_MODEL}" \
  --whispervq-model "${WHISPERVQ_MODEL}" \
  --bicodec-checkpoint "${BICODEC_CHECKPOINT}" \
  --rank 0 \
  --world-size 15 \
  --shard-count 15 \
  --batch-size "${SMOKE_BATCH_SIZE:-4}" \
  --teacher-request-batch-size "${SMOKE_TEACHER_BATCH_SIZE:-64}" \
  --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
  --max-rows-per-shard "${SMOKE_ROWS:-4}" \
  --progress-interval 1 2>&1 | tee "${LOG_ROOT}/smoke/cache_1gpu.log"

"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import TrajectoryRecord
root=Path(os.environ["SMOKE_ROOT"])/"part-000"
rows=[TrajectoryRecord.from_dict(json.loads(line)) for line in (root/"trajectory_cache.jsonl").read_text().splitlines() if line]
assert rows
for sample_id in sorted({row.sample_id for row in rows}):
    session=[row for row in rows if row.sample_id==sample_id]
    if session[0].source_duration_ms >= 800:
        assert any(row.chunk_end_ms == 800 for row in session)
print({"smoke_records":len(rows),"smoke_sessions":len({row.sample_id for row in rows})})
PY
