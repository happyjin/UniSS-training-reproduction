#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

refuse_existing "${PILOT_ROOT}" "${PILOT_REPLAY_OFFSETS}" "${PILOT_REPLAY_OFFSETS}.json"
require_file "${UNIST_DEV_STAGE_A}"
require_file "${FULL_REPLAY_OFFSETS}"
require_file "${FULL_REPLAY_OFFSETS}.json"

TRAIN_ARGS=()
for shard in $(seq 0 14); do
  manifest="$(printf '%s/parts/train-%05d/manifest.jsonl' "${FULL_STAGE_A_ROOT}" "${shard}")"
  require_file "${manifest}"
  TRAIN_ARGS+=(--train-source "${manifest}")
done

"${PYTHON}" -m training.phase3_whisper_streamspeech_joint.build_joint_manifests_parallel \
  "${TRAIN_ARGS[@]}" \
  --valid-source "${UNIST_DEV_STAGE_A}" \
  --output-dir "${PILOT_ROOT}" \
  --phase3-model "${PHASE3_MODEL}" \
  --workers "${JOINT_MANIFEST_WORKERS:-16}" \
  --validation-per-mille 0 \
  --skip-audio-check \
  --skip-empty-target-bicodec

mkdir -p "$(dirname "${PILOT_REPLAY_OFFSETS}")"
"${PYTHON}" - "${FULL_REPLAY_OFFSETS}" "${FULL_REPLAY_OFFSETS}.json" \
  "${PILOT_REPLAY_OFFSETS}" "${PILOT_REPLAY_RECORDS:-100000}" <<'PY'
import array
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
source_meta = json.loads(Path(sys.argv[2]).read_text())
output = Path(sys.argv[3]).resolve()
count = int(sys.argv[4])
values = array.array("Q")
with source.open("rb") as handle:
    values.fromfile(handle, min(count, source.stat().st_size // values.itemsize))
if not values:
    raise SystemExit("full replay index is empty")
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
try:
    with os.fdopen(descriptor, "wb") as handle:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
finally:
    Path(temporary_name).unlink(missing_ok=True)
metadata = dict(source_meta)
metadata.update(
    offsets=str(output), records=len(values), complete=False, max_records=len(values)
)
Path(f"{output}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
print(json.dumps(metadata, sort_keys=True))
PY
