#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

[[ -x "${ENV_ROOT}/bin/python" ]] || { echo "Missing environment: ${ENV_ROOT}" >&2; exit 1; }
[[ -f "${EVAL_SOURCE_PARQUET}" ]] || { echo "Missing eval parquet: ${EVAL_SOURCE_PARQUET}" >&2; exit 1; }
[[ -d "${TOKENIZER}" ]] || { echo "Missing tokenizer: ${TOKENIZER}" >&2; exit 1; }
[[ -f "${DEV_SAMPLES}" && -f "${DEV_SCHEDULES}" ]] || { echo "Missing dev inputs" >&2; exit 1; }

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MARKER="${EVAL_DATA_DIR}/PREPARE_COMPLETE.json"
if [[ -f "${MARKER}" ]]; then
  "${ENV_ROOT}/bin/python" - "${MARKER}" "${EXPECTED_EVAL_SAMPLES}" <<'PY'
import json, pathlib, sys
marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
if marker.get("records") != expected:
    raise SystemExit(f"marker records {marker.get('records')} != {expected}")
for field in ("schedules", "samples", "action_samples", "stats", "manifest"):
    path = pathlib.Path(marker[field])
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing prepared artifact: {path}")
print(json.dumps(marker, sort_keys=True))
PY
  echo "Reusing prepared eval data: ${EVAL_DATA_DIR}"
  exit 0
fi

if [[ -e "${EVAL_DATA_DIR}" ]]; then
  echo "Refusing to overwrite incomplete eval data directory: ${EVAL_DATA_DIR}" >&2
  exit 1
fi

PARENT="$(dirname "${EVAL_DATA_DIR}")"
mkdir -p "${PARENT}"
PARTIAL="${EVAL_DATA_DIR}.partial.$$"
cleanup() {
  if [[ -d "${PARTIAL}" ]]; then
    rm -rf -- "${PARTIAL}"
  fi
}
trap cleanup EXIT
mkdir -p "${PARTIAL}"

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/training/simul_uniss/prepare_data.py" \
  --input "${EVAL_SOURCE_PARQUET}" \
  --output-dir "${PARTIAL}" \
  --tokenizer "${TOKENIZER}" \
  --chunk-ms "${CHUNK_MS}" \
  --wait-k-chunks "${WAIT_K_CHUNKS}" \
  --max-phrase-tokens "${MAX_PHRASE_TOKENS}" \
  --progress-interval 2000

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/training/simul_uniss/mask_action_samples.py" \
  --input "${PARTIAL}/samples.jsonl" \
  --output "${PARTIAL}/action_samples.jsonl"

"${ENV_ROOT}/bin/python" - "${PARTIAL}" "${EVAL_DATA_DIR}" "${EXPECTED_EVAL_SAMPLES}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
final_root = pathlib.Path(sys.argv[2]).resolve()
expected = int(sys.argv[3])
stats = json.loads((root / "stats.json").read_text())
if stats.get("records") != expected:
    raise SystemExit(f"prepared records {stats.get('records')} != {expected}")
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
files = {name: root / name for name in (
    "schedules.jsonl", "samples.jsonl", "action_samples.jsonl", "stats.json", "manifest.json"
)}
marker = {
    "schema_version": "simul_uniss_stage3_eval_data_v1",
    "records": expected,
    **{
        key.removesuffix(".jsonl").removesuffix(".json"): str(final_root / key)
        for key in files
    },
    "sha256": {key: sha256(path) for key, path in files.items()},
}
(root / "PREPARE_COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n")
PY

mv "${PARTIAL}" "${EVAL_DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
