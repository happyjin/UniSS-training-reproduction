#!/usr/bin/env bash
set -euo pipefail
DRY_RUN=0
SHARD_INDEX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --index) SHARD_INDEX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "--index is required" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
printf -v name 'train-%05d' "${SHARD_INDEX}"
source_samples="${SOURCE_PREPARED_PARTS_DIR}/${name}/samples.jsonl"
output_dir="${PACKED_PARTS_DIR}/${name}"
output="${output_dir}/packed_interleaved.jsonl"
marker="${output_dir}/PACK_INTERLEAVED_COMPLETE.json"
log="${LOG_DIR}/data_preparation/pack_${name}.log"
cmd=(python -m training.simul_uniss.repack_interleaved pack
  --input "${source_samples}" --output "${output}" --marker "${marker}"
  --seq-length "${SEQ_LENGTH}")
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
mkdir -p "${output_dir}" "$(dirname "${log}")"
"${cmd[@]}" 2>&1 | tee -a "${log}"
