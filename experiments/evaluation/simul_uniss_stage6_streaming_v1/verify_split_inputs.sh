#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "dev" && "$1" != "test" ) ]]; then
  echo "Usage: $0 dev|test" >&2
  exit 2
fi
SPLIT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

if [[ "${SPLIT}" == "dev" ]]; then
  SCHEDULES="${DEV_SCHEDULES}"
  MANIFEST="${DEV_MANIFEST}"
  EXPECTED="${EXPECTED_DEV_RECORDS}"
else
  SCHEDULES="${TEST_SCHEDULES}"
  MANIFEST="${TEST_MANIFEST}"
  EXPECTED="${EXPECTED_TEST_RECORDS}"
fi

"${TRAIN_ENV}/bin/python" - "${SCHEDULES}" "${MANIFEST}" "${EXPECTED}" "${SPLIT}" <<'PY'
import json
import pathlib
import sys

schedules_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
expected = int(sys.argv[3])
split = sys.argv[4]
if not schedules_path.is_file() or not manifest_path.is_file():
    raise SystemExit(f"missing {split} schedule or offline manifest")
schedules = [json.loads(line) for line in schedules_path.open(encoding="utf-8") if line.strip()]
manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
if len(schedules) != expected or len(manifest) != expected:
    raise SystemExit(
        f"{split} count mismatch schedules={len(schedules)} manifest={len(manifest)} expected={expected}"
    )
schedule_ids = [str(row["id"]) for row in schedules]
manifest_ids = [str(row["id"]) for row in manifest]
if schedule_ids != manifest_ids:
    mismatch = next(
        (index for index, pair in enumerate(zip(schedule_ids, manifest_ids)) if pair[0] != pair[1]),
        None,
    )
    raise SystemExit(f"{split} schedule/manifest ID order mismatch at {mismatch}")
if len(set(schedule_ids)) != expected:
    raise SystemExit(f"duplicate {split} IDs")
required = {"id", "events", "speaker_tokens", "src_lang", "tgt_lang", "translation"}
if any(not required.issubset(row) or not row["events"] for row in schedules):
    raise SystemExit(f"incomplete {split} schedule records")
wrong_split = [row["id"] for row in schedules if row.get("split", split) != split]
if wrong_split:
    raise SystemExit(f"{split} schedule contains wrong split IDs: {wrong_split[:3]}")
print(json.dumps({
    "split": split,
    "records": expected,
    "first_id": schedule_ids[0],
    "last_id": schedule_ids[-1],
}, ensure_ascii=False, sort_keys=True))
PY
