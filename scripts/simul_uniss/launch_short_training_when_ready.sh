#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
POLL_SECONDS=60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

GPU_SMOKE_MARKER="${RUN_DIR}/gpu_smoke/GPU_SMOKE_COMPLETE"
SHORT_TRAINING_MARKER="${RUN_DIR}/short_training/SHORT_TRAINING_COMPLETE"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'wait for %s\n' "${GPU_SMOKE_MARKER}"
  printf 'wait for %s\n' "${ACTION_PREPARE_MARKER}"
  printf 'then run: %s --config %s\n' \
    "${REPO_ROOT}/scripts/simul_uniss/run_short_training_pipeline.sh" "${CONFIG_FILE}"
  exit 0
fi

if [[ -f "${SHORT_TRAINING_MARKER}" ]]; then
  echo "Short training is already complete: ${SHORT_TRAINING_MARKER}"
  exit 0
fi

while [[ ! -f "${GPU_SMOKE_MARKER}" || ! -f "${ACTION_PREPARE_MARKER}" ]]; do
  printf '[%s] waiting for prerequisites: gpu_smoke=%s action_data=%s\n' \
    "$(date -u +%FT%TZ)" "$([[ -f "${GPU_SMOKE_MARKER}" ]] && echo ready || echo pending)" \
    "$([[ -f "${ACTION_PREPARE_MARKER}" ]] && echo ready || echo pending)"
  sleep "${POLL_SECONDS}"
done

exec "${REPO_ROOT}/scripts/simul_uniss/run_short_training_pipeline.sh" --config "${CONFIG_FILE}"
