#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
START_PHASE="${START_PHASE:-phase1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --start-phase) START_PHASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_full_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

case "${START_PHASE}" in
  phase1|phase2|phase3) ;;
  *) echo "START_PHASE must be phase1, phase2, or phase3" >&2; exit 2 ;;
esac

if [[ "${DRY_RUN}" == "0" ]]; then
  if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
    echo "Missing activation script: ${ACTIVATE_SCRIPT}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${ACTIVATE_SCRIPT}"
fi

should_run() {
  local phase="$1"
  case "${START_PHASE}:${phase}" in
    phase1:*) return 0 ;;
    phase2:phase1) return 1 ;;
    phase2:*) return 0 ;;
    phase3:phase3) return 0 ;;
    *) return 1 ;;
  esac
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
}

validate_existing_packed() {
  local output="$1"
  local count_file="${output}.count"
  [[ -s "${output}" && -s "${count_file}" ]] || return 1
  [[ "$(<"${count_file}")" =~ ^[1-9][0-9]*$ ]] || return 1
  python "${REPO_ROOT}/training/validate_packed_jsonl.py" \
    --input "${output}" --seq-length "${SEQ_LENGTH}" >/dev/null
}

pack_one() {
  local label="$1"
  local output="$2"
  shift 2
  local inputs=("$@")
  local tmp="${output}.tmp.$$"
  local report="${PACK_RUN_DIR}/${label}_packer_report.$$.json"
  local count_file="${output}.count"
  local preserved
  local workers="${PACK_WORKERS:-1}"
  if [[ "${label}" == *validation* ]]; then
    workers=1
  fi
  local packer=(python "${REPO_ROOT}/training/pack_sequences.py")
  if (( workers > 1 )); then
    packer=(python "${REPO_ROOT}/training/pack_sequences_parallel.py" --workers "${workers}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] pack ${label} with ${workers} worker(s) -> ${output}"
    print_cmd "${packer[@]}" \
      --input "${inputs[@]}" --output "${tmp}" \
      --seq-length "${SEQ_LENGTH}" --drop-overlong
    print_cmd python "${REPO_ROOT}/training/validate_packed_jsonl.py" \
      --input "${tmp}" --seq-length "${SEQ_LENGTH}"
    echo "[dry-run] atomic mv ${tmp} -> ${output}; write ${count_file}"
    return 0
  fi

  if validate_existing_packed "${output}"; then
    echo "[$(date -u +%FT%TZ)] ${label} already complete: ${output} ($( <"${count_file}" ) records)"
    return 0
  fi

  mkdir -p "$(dirname "${output}")" "${PACK_RUN_DIR}"
  echo "[$(date -u +%FT%TZ)] packing ${label} with ${workers} worker(s); temporary output: ${tmp}"
  "${packer[@]}" \
    --input "${inputs[@]}" --output "${tmp}" \
    --seq-length "${SEQ_LENGTH}" --drop-overlong | tee "${report}"

  local reported_count actual_count
  reported_count="$(python - "${report}" <<'PY'
import json
import sys
from pathlib import Path

lines = [line for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.loads(lines[-1])["packed_sequences"])
PY
)"
  actual_count="$(wc -l < "${tmp}")"
  if [[ "${reported_count}" != "${actual_count}" || "${actual_count}" == "0" ]]; then
    echo "${label} count mismatch: packer=${reported_count}, wc=${actual_count}; preserving ${tmp}" >&2
    exit 1
  fi

  python "${REPO_ROOT}/training/validate_packed_jsonl.py" \
    --input "${tmp}" --seq-length "${SEQ_LENGTH}"
  if [[ -e "${output}" ]]; then
    preserved="${output}.invalid.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "${output}" "${preserved}"
    echo "Preserved previous invalid output as ${preserved}"
  fi
  mv "${tmp}" "${output}"
  printf '%s\n' "${actual_count}" > "${count_file}.tmp.$$"
  mv "${count_file}.tmp.$$" "${count_file}"
  echo "[$(date -u +%FT%TZ)] completed ${label}: ${output} (${actual_count} records)"
}

validate_phase3_source() {
  python - "${PHASE3_DEV_SOURCE}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

counts = Counter()
with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    for line in handle:
        item = json.loads(line)
        counts[item["task"]] += 1
if not counts or set(counts) != {"quality", "performance"}:
    raise SystemExit(f"invalid Phase3 validation task counts: {dict(counts)}")
if counts["quality"] != counts["performance"]:
    raise SystemExit(f"unbalanced Phase3 validation tasks: {dict(counts)}")
print(json.dumps(counts, sort_keys=True))
PY
}

prepare_phase3_validation() {
  local source_tmp="${PHASE3_DEV_SOURCE}.tmp.$$"
  local packed_tmp="${PHASE3_VALID}.tmp.$$"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] build isolated Phase3 validation"
    print_cmd python "${REPO_ROOT}/training/prepare_unist_s2st.py" \
      --input "${UNIST_DEV_PARQUET}" --phase phase3 \
      --tokenizer "${TOKENIZER_DIR}" --output "${source_tmp}"
    print_cmd python "${REPO_ROOT}/training/pack_sequences.py" \
      --input "${PHASE3_DEV_SOURCE}" --output "${packed_tmp}" \
      --seq-length "${SEQ_LENGTH}" --drop-overlong
    return 0
  fi

  if [[ ! -s "${PHASE3_DEV_SOURCE}" ]]; then
    require_file "${UNIST_DEV_PARQUET}"
    mkdir -p "$(dirname "${PHASE3_DEV_SOURCE}")"
    python "${REPO_ROOT}/training/prepare_unist_s2st.py" \
      --input "${UNIST_DEV_PARQUET}" --phase phase3 \
      --tokenizer "${TOKENIZER_DIR}" --output "${source_tmp}"
    mv "${source_tmp}" "${PHASE3_DEV_SOURCE}"
  fi
  validate_phase3_source

  if ! validate_existing_packed "${PHASE3_VALID}"; then
    pack_one "phase3_validation" "${PHASE3_VALID}" "${PHASE3_DEV_SOURCE}"
  fi
}

if [[ "${DRY_RUN}" == "0" && -f "${PACKING_COMPLETE_MARKER}" && "${START_PHASE}" == "phase1" ]]; then
  marker_valid=1
  for path in "${PHASE1_TRAIN}" "${PHASE2_TRAIN}" "${PHASE3_TRAIN}" "${PHASE3_VALID}"; do
    if ! validate_existing_packed "${path}"; then
      marker_valid=0
      break
    fi
  done
  if [[ "${marker_valid}" == "1" ]]; then
    echo "Packing completion marker and artifacts are valid: ${PACKING_COMPLETE_MARKER}"
    exit 0
  fi
  echo "Packing marker exists but at least one artifact is invalid; resuming validation/packing"
fi

if should_run phase1; then
  mapfile -d '' PHASE1_INPUTS < <(find "${PHASE1_SOURCE_DIR}" -maxdepth 1 -type f -name 'train-*.jsonl' -print0 | sort -z)
  if [[ "${#PHASE1_INPUTS[@]}" -ne 198 ]]; then
    echo "Expected 198 Phase1 shards, found ${#PHASE1_INPUTS[@]} in ${PHASE1_SOURCE_DIR}" >&2
    exit 1
  fi
  pack_one phase1 "${PHASE1_TRAIN}" "${PHASE1_INPUTS[@]}"
fi

if should_run phase2; then
  require_file "${PHASE2_SOURCE}"
  pack_one phase2 "${PHASE2_TRAIN}" "${PHASE2_SOURCE}"
fi

if should_run phase3; then
  mapfile -d '' PHASE3_INPUTS < <(find "${PHASE3_SOURCE_DIR}" -maxdepth 1 -type f -name 'train-*.jsonl' -print0 | sort -z)
  if [[ "${#PHASE3_INPUTS[@]}" -ne 198 ]]; then
    echo "Expected 198 Phase3 shards, found ${#PHASE3_INPUTS[@]} in ${PHASE3_SOURCE_DIR}" >&2
    exit 1
  fi
  pack_one phase3 "${PHASE3_TRAIN}" "${PHASE3_INPUTS[@]}"
  prepare_phase3_validation
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p "${PACK_RUN_DIR}"
  marker_tmp="${PACKING_COMPLETE_MARKER}.tmp.$$"
  {
    echo "completed_at=$(date -u +%FT%TZ)"
    echo "seq_length=${SEQ_LENGTH}"
    for path in "${PHASE1_TRAIN}" "${PHASE2_TRAIN}" "${PHASE3_TRAIN}" "${PHASE3_VALID}"; do
      require_file "${path}"
      require_file "${path}.count"
      echo "artifact=${path} count=$(<"${path}.count") bytes=$(stat -c %s "${path}") mtime=$(stat -c %y "${path}")"
    done
  } > "${marker_tmp}"
  mv "${marker_tmp}" "${PACKING_COMPLETE_MARKER}"
  echo "Packing complete: ${PACKING_COMPLETE_MARKER}"
fi
