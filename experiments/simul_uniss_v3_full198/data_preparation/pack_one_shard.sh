#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
SHARD_INDEX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --index) SHARD_INDEX="$2"; shift 2 ;;
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
prepared_part="${PREPARED_PARTS_DIR}/${name}"
packed_part="${PACKED_PARTS_DIR}/${name}"
marker="${packed_part}/PACK_COMPLETE.json"
temporary="${PACKED_PARTS_DIR}/.${name}.tmp.$$"
action_samples="${temporary}/action_samples.jsonl"
log_file="${LOG_DIR}/data_preparation/pack_${name}.log"

interleaved_cmd=(python -m training.simul_uniss.pack_sequences
  --input "${prepared_part}/samples.jsonl"
  --output "${temporary}/packed_interleaved.jsonl"
  --seq-length "${SEQ_LENGTH}" --drop-overlong)
mask_cmd=(python -m training.simul_uniss.mask_action_samples
  --input "${prepared_part}/samples.jsonl" --output "${action_samples}")
action_cmd=(python -m training.simul_uniss.pack_sequences
  --input "${action_samples}" --output "${temporary}/packed_action.jsonl"
  --seq-length "${SEQ_LENGTH}" --drop-overlong)
mark_cmd=(python -m training.simul_uniss.full_data_pipeline mark-packed
  --prepared-part "${prepared_part}" --packed-part "${temporary}" --shard-index "${SHARD_INDEX}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${interleaved_cmd[@]}"; printf '\n'
  printf '%q ' "${mask_cmd[@]}"; printf '\n'
  printf '%q ' "${action_cmd[@]}"; printf '\n'
  printf '%q ' "${mark_cmd[@]}"; printf '\n'
  printf 'atomic_publish %q %q\n' "${temporary}" "${packed_part}"
  exit 0
fi
[[ -f "${prepared_part}/PREPARE_COMPLETE.json" ]] || {
  echo "Prepared part incomplete: ${prepared_part}" >&2; exit 1;
}
if [[ -f "${marker}" ]]; then
  python -m training.simul_uniss.full_data_pipeline verify-packed \
    --prepared-part "${prepared_part}" --packed-part "${packed_part}" --shard-index "${SHARD_INDEX}" >/dev/null
  echo "Skipping verified packed part: ${packed_part}"
  exit 0
fi
[[ ! -e "${packed_part}" ]] || { echo "Refusing partial/non-marked part: ${packed_part}" >&2; exit 1; }
mkdir -p "${PACKED_PARTS_DIR}" "$(dirname "${log_file}")"
rm -rf -- "${temporary}"
cleanup() { rm -rf -- "${temporary}"; }
trap cleanup EXIT
{
  mkdir -p "${temporary}"
  "${interleaved_cmd[@]}"
  "${mask_cmd[@]}"
  "${action_cmd[@]}"
  rm -f "${action_samples}"
  "${mark_cmd[@]}"
  mv "${temporary}" "${packed_part}"
  echo "Packed ${name}"
} 2>&1 | tee -a "${log_file}"
