#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
RETRY_SECONDS="${RETRY_SECONDS:-30}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --retry-seconds) RETRY_SECONDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_ab.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

mkdir -p "${LOG_ROOT}"
pipeline_log="${LOG_ROOT}/stage_ab_pipeline.log"

run_until_complete() {
  local stage_name="$1"
  local completion_marker="$2"
  shift 2
  while [[ ! -f "${completion_marker}" ]]; do
    echo "[$(date -u +%FT%TZ)] Starting ${stage_name}" | tee -a "${pipeline_log}"
    if "$@"; then
      if [[ -f "${completion_marker}" ]]; then
        echo "[$(date -u +%FT%TZ)] ${stage_name} complete" | tee -a "${pipeline_log}"
        return 0
      fi
      echo "[$(date -u +%FT%TZ)] ${stage_name} exited without completion marker" \
        | tee -a "${pipeline_log}"
    else
      status=$?
      echo "[$(date -u +%FT%TZ)] ${stage_name} failed with status ${status}" \
        | tee -a "${pipeline_log}"
    fi
    echo "[$(date -u +%FT%TZ)] Retrying ${stage_name} in ${RETRY_SECONDS}s" \
      | tee -a "${pipeline_log}"
    sleep "${RETRY_SECONDS}"
  done
}

run_until_complete \
  "Stage A source preparation" \
  "${STAGE_A_ROOT}/STAGE_A_SOURCE_COMPLETE.json" \
  "${REPO_ROOT}/scripts/simul_uniss_subsecond_v1/prepare_stage_a_pilot.sh" \
  --config "${CONFIG_FILE}" --formal

run_until_complete \
  "Stage B eight-GPU training" \
  "${STAGE_B_ROOT}/STAGE_B_PILOT_COMPLETE.json" \
  "${REPO_ROOT}/scripts/simul_uniss_subsecond_v1/train_stage_b.sh" \
  --config "${CONFIG_FILE}" --formal --resume

echo "[$(date -u +%FT%TZ)] Stage A/B pipeline complete" | tee -a "${pipeline_log}"
