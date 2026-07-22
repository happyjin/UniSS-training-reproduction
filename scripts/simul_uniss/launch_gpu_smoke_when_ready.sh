#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
POLL_SECONDS=60
TARGET_ITERATION=15465
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --target-iteration) TARGET_ITERATION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

PHASE1_ITERATION_FILE="${QWEN_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
GPU_SMOKE_MARKER="${RUN_DIR}/gpu_smoke/GPU_SMOKE_COMPLETE"
# The bracketed first character keeps pgrep from matching this launcher's own
# command line while still matching Python workers running the legacy trainer.
PHASE1_PROCESS_PATTERN='[p]ython[^ ]* .*training/pretrain_uniss_megatron\.py'

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'wait for iteration >= %s in %s\n' "${TARGET_ITERATION}" "${PHASE1_ITERATION_FILE}"
  printf 'wait while process pattern is active: %s\n' "${PHASE1_PROCESS_PATTERN}"
  printf 'then run: %s --config %s\n' \
    "${REPO_ROOT}/scripts/simul_uniss/run_gpu_smoke_pipeline.sh" "${CONFIG_FILE}"
  exit 0
fi

if [[ -f "${GPU_SMOKE_MARKER}" ]]; then
  echo "GPU smoke is already complete: ${GPU_SMOKE_MARKER}"
  exit 0
fi

while true; do
  iteration="$(tr -d '[:space:]' < "${PHASE1_ITERATION_FILE}" 2>/dev/null || echo 0)"
  phase1_active=0
  if pgrep -f "${PHASE1_PROCESS_PATTERN}" >/dev/null; then
    phase1_active=1
  fi
  printf '[%s] waiting for Phase1: iteration=%s target=%s active=%s\n' \
    "$(date -u +%FT%TZ)" "${iteration:-0}" "${TARGET_ITERATION}" "${phase1_active}"
  if [[ "${iteration:-0}" =~ ^[0-9]+$ ]] \
      && (( iteration >= TARGET_ITERATION )) \
      && (( phase1_active == 0 )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

exec "${REPO_ROOT}/scripts/simul_uniss/run_gpu_smoke_pipeline.sh" --config "${CONFIG_FILE}"
