#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
MODE="formal"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --smoke) MODE="smoke"; shift ;;
    --formal) MODE="formal"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_ab.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

if [[ "${MODE}" == "smoke" ]]; then
  output_root="${STAGE_A_SMOKE_ROOT}"
  records_per_shard="${STAGE_A_SMOKE_RECORDS_PER_SHARD}"
  shard_count="${SHARD_COUNT}"
  skip_sha=(--skip-sha256)
else
  output_root="${STAGE_A_ROOT}"
  records_per_shard="${STAGE_A_RECORDS_PER_SHARD}"
  shard_count="${SHARD_COUNT}"
  skip_sha=()
fi

mkdir -p "${output_root}" "${LOG_ROOT}/stage_a"
echo "[$(date -u +%FT%TZ)] Stage A ${MODE}: ${shard_count} shards, ${records_per_shard} records/shard" \
  | tee -a "${LOG_ROOT}/stage_a/${MODE}_launcher.log"

for ((wave_start=0; wave_start<shard_count; wave_start+=8)); do
  pids=()
  for ((slot=0; slot<8 && wave_start+slot<shard_count; slot++)); do
    shard_index=$((SHARD_START + wave_start + slot))
    "${REPO_ROOT}/scripts/simul_uniss_subsecond_v1/run_stage_a_part.sh" \
      --config "${CONFIG_FILE}" \
      --shard-index "${shard_index}" \
      --gpu "${slot}" \
      --output-root "${output_root}" \
      --limit-records "${records_per_shard}" \
      --side "${STAGE_A_SIDE}" \
      "${skip_sha[@]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

python -m training.simul_uniss.subsecond_v1.stage_a assemble \
  --parts-root "${output_root}/parts" \
  --output-dir "${output_root}" \
  --shard-start "${SHARD_START}" \
  --shard-count "${shard_count}" \
  2>&1 | tee -a "${LOG_ROOT}/stage_a/${MODE}_assemble.log"

python -m training.simul_uniss.subsecond_v1.stage_a validate \
  --output-dir "${output_root}" \
  --samples 32 \
  2>&1 | tee -a "${LOG_ROOT}/stage_a/${MODE}_validate.log"

echo "[$(date -u +%FT%TZ)] Stage A ${MODE} complete: ${output_root}" \
  | tee -a "${LOG_ROOT}/stage_a/${MODE}_launcher.log"
