#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
"${TRAIN_ENV}/bin/python" - "${DEV_SCHEDULES}" "${OFFLINE_DEV_MANIFEST}" "${EXPECTED_DEV_RECORDS}" <<'PY'
import json
import pathlib
import sys

schedules, manifest = map(pathlib.Path, sys.argv[1:3])
expected = int(sys.argv[3])
if not schedules.is_file() or not manifest.is_file():
    raise SystemExit("missing Stage4 dev schedule or offline dev manifest")
schedule_ids = [str(json.loads(line)["id"]) for line in schedules.open() if line.strip()]
manifest_ids = [str(json.loads(line)["id"]) for line in manifest.open() if line.strip()]
if len(schedule_ids) != expected or len(manifest_ids) != expected:
    raise SystemExit(
        f"record count mismatch schedules={len(schedule_ids)} "
        f"manifest={len(manifest_ids)} expected={expected}"
    )
if schedule_ids != manifest_ids:
    for index, (left, right) in enumerate(zip(schedule_ids, manifest_ids)):
        if left != right:
            raise SystemExit(f"id order mismatch at {index}: {left!r} != {right!r}")
    raise SystemExit("schedule/manifest ids differ")
if len(set(schedule_ids)) != expected:
    raise SystemExit("duplicate dev ids")
print(json.dumps({"records": expected, "first_id": schedule_ids[0], "last_id": schedule_ids[-1]}))
PY
