#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
"${BASE_STAGE4_EVAL}/verify_dev_inputs.sh"
"${TRAIN_ENV}/bin/python" - "${TEST_SCHEDULES}" "${TEST_PROCESSED_DIR}/manifest.json" "${EXPECTED_TEST_RECORDS}" <<'PY'
import json
import pathlib
import statistics
import sys

schedules_path, manifest_path = map(pathlib.Path, sys.argv[1:3])
expected = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("chunk_ms") != 640 or manifest.get("wait_k_chunks") != 2:
    raise SystemExit("test schedule operating point differs from frozen dev configuration")
if manifest.get("max_phrase_tokens") != 16:
    raise SystemExit("test max_phrase_tokens differs from frozen dev configuration")
rows = [json.loads(line) for line in schedules_path.open(encoding="utf-8") if line.strip()]
if len(rows) != expected or any(row.get("split") != "test" for row in rows):
    raise SystemExit("test schedule count/split validation failed")
required = {"id", "events", "speaker_tokens", "src_lang", "tgt_lang", "translation"}
if any(not required.issubset(row) or not row["events"] for row in rows):
    raise SystemExit("test schedule contains incomplete records")
events = [len(row["events"]) for row in rows]
worst_context = []
for row in rows:
    source_tokens = sum(len(event["source_glm"]) + 4 for event in row["events"])
    worst_context.append(64 + source_tokens + 700 * len(row["events"]))
if max(worst_context) >= 32768:
    raise SystemExit("conservative free-running context can exceed native Qwen context")
directions = {}
for row in rows:
    key = f"{row['src_lang']}->{row['tgt_lang']}"
    directions[key] = directions.get(key, 0) + 1
print(json.dumps({
    "records": len(rows),
    "events": sum(events),
    "events_mean": statistics.mean(events),
    "events_max": max(events),
    "direction_counts": directions,
    "conservative_context_max": max(worst_context),
    "first_id": rows[0]["id"],
    "last_id": rows[-1]["id"],
}, ensure_ascii=False, sort_keys=True))
PY
