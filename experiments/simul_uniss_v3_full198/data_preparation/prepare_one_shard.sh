#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
SHARD_INDEX=""
LIMIT_RECORDS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --index) SHARD_INDEX="$2"; shift 2 ;;
    --limit-records) LIMIT_RECORDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "--index is required" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/experiment.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

(( SHARD_INDEX >= SHARD_START && SHARD_INDEX < SHARD_START + SHARD_COUNT )) || {
  echo "Shard ${SHARD_INDEX} is outside [${SHARD_START}, $((SHARD_START + SHARD_COUNT)))" >&2
  exit 2
}
printf -v name 'train-%05d' "${SHARD_INDEX}"
source_parquet="${UNIST_ROOT}/${name}.parquet"
part_dir="${PREPARED_PARTS_DIR}/${name}"
marker="${part_dir}/PREPARE_COMPLETE.json"
temporary="${PREPARED_PARTS_DIR}/.${name}.tmp.$$"
log_file="${LOG_DIR}/data_preparation/prepare_${name}.log"

prepare_cmd=(python -m training.simul_uniss.prepare_data
  --input "${source_parquet}"
  --output-dir "${temporary}"
  --tokenizer "${TOKENIZER_DIR}"
  --chunk-ms "${CHUNK_MS}"
  --wait-k-chunks "${WAIT_K_CHUNKS}"
  --max-phrase-tokens "${MAX_PHRASE_TOKENS}"
  --skip-invalid-records)
if [[ -n "${LIMIT_RECORDS}" ]]; then
  prepare_cmd+=(--limit-records "${LIMIT_RECORDS}")
fi
mark_cmd=(python -m training.simul_uniss.full_data_pipeline mark-prepared
  --source "${source_parquet}" --part-dir "${temporary}" --published-dir "${part_dir}"
  --shard-index "${SHARD_INDEX}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${prepare_cmd[@]}"; printf '\n'
  printf '%q ' "${mark_cmd[@]}"; printf '\n'
  printf 'atomic_publish %q %q\n' "${temporary}" "${part_dir}"
  exit 0
fi
[[ -f "${source_parquet}" ]] || { echo "Missing source shard: ${source_parquet}" >&2; exit 1; }
if [[ -f "${marker}" ]]; then
  python -m training.simul_uniss.full_data_pipeline verify-prepared \
    --source "${source_parquet}" --part-dir "${part_dir}" --shard-index "${SHARD_INDEX}" >/dev/null
  echo "Skipping verified prepared part: ${part_dir}"
  exit 0
fi
[[ ! -e "${part_dir}" ]] || { echo "Refusing partial/non-marked part: ${part_dir}" >&2; exit 1; }
mkdir -p "${PREPARED_PARTS_DIR}" "$(dirname "${log_file}")"
rm -rf -- "${temporary}"
cleanup() { rm -rf -- "${temporary}"; }
trap cleanup EXIT
{
  "${prepare_cmd[@]}"
  "${mark_cmd[@]}"
  mv "${temporary}" "${part_dir}"
  echo "Prepared ${name}"
} 2>&1 | tee -a "${log_file}"
