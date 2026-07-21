#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_full_v1.env}"
PACK_START_PHASE="${PACK_START_PHASE:-phase1}"
TRAIN_START_PHASE="${TRAIN_START_PHASE:-phase1}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q --config %q --start-phase %q\n' "${REPO_ROOT}/scripts/pack_unist198_full.sh" "${CONFIG_FILE}" "${PACK_START_PHASE}"
  printf '%q --config %q --start-phase %q\n' "${REPO_ROOT}/scripts/run_qwen0p5b_unist198_all_phases.sh" "${CONFIG_FILE}" "${TRAIN_START_PHASE}"
  exit 0
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"
{
  echo "[$(date -u +%FT%TZ)] starting full UniST-198 packing and training pipeline"
  "${REPO_ROOT}/scripts/pack_unist198_full.sh" --config "${CONFIG_FILE}" --start-phase "${PACK_START_PHASE}"
  "${REPO_ROOT}/scripts/run_qwen0p5b_unist198_all_phases.sh" --config "${CONFIG_FILE}" --start-phase "${TRAIN_START_PHASE}"
  echo "[$(date -u +%FT%TZ)] full UniST-198 pipeline completed"
} 2>&1 | tee -a "${PIPELINE_LOG}"
