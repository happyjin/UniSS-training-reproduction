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
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

PACK_SESSION="${PACK_SESSION:-unist198_full_v1}"
PHASE1_SESSION="${PHASE1_SESSION:-unist198_phase1_train}"
PHASE23_SESSION="${PHASE23_SESSION:-unist198_phase23_train}"
POLL_SECONDS="${POLL_SECONDS:-1}"
GUARD_LOG="${GUARD_LOG:-${REPO_ROOT}/logs/unist198_stage_guard.log}"
PHASE1_TRACKER="${PHASE1_SAVE}/latest_checkpointed_iteration.txt"

phase1_count="${PHASE1_PACKED_COUNT_OVERRIDE:-}"
if [[ -z "${phase1_count}" ]]; then
  [[ -s "${PHASE1_TRAIN}.count" ]] || { echo "Missing Phase1 count: ${PHASE1_TRAIN}.count" >&2; exit 1; }
  phase1_count="$(<"${PHASE1_TRAIN}.count")"
fi
[[ "${phase1_count}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid Phase1 count: ${phase1_count}" >&2; exit 1; }
phase1_epoch_iters=$(( (phase1_count + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
phase1_target="${PHASE1_TRAIN_ITERS:-$((3 * phase1_epoch_iters))}"

tracker_iteration() {
  if [[ ! -s "${PHASE1_TRACKER}" ]]; then
    echo -1
    return
  fi
  local value
  value="$(tr -d '[:space:]' < "${PHASE1_TRACKER}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint tracker: ${PHASE1_TRACKER}" >&2; exit 1; }
  echo "${value}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] wait for ${PACKING_COMPLETE_MARKER} while ${PACK_SESSION} and ${PHASE1_SESSION} run in parallel"
  echo "[dry-run] stop ${PACK_SESSION} after the atomic packing marker appears"
  echo "[dry-run] wait for Phase1 iteration ${phase1_target}, then start ${PHASE23_SESSION} at phase2"
  exit 0
fi

mkdir -p "$(dirname "${GUARD_LOG}")"
log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${GUARD_LOG}"
}

log "guard started: packing=${PACK_SESSION}, phase1=${PHASE1_SESSION}, target=${phase1_target}"
while [[ ! -f "${PACKING_COMPLETE_MARKER}" ]]; do
  if ! tmux has-session -t "${PACK_SESSION}" 2>/dev/null; then
    log "ERROR packing session exited before ${PACKING_COMPLETE_MARKER} appeared"
    exit 1
  fi
  if ! tmux has-session -t "${PHASE1_SESSION}" 2>/dev/null \
      && [[ "$(tracker_iteration)" != "${phase1_target}" ]]; then
    log "ERROR Phase1 session exited at iteration $(tracker_iteration)"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

log "packing marker appeared; stopping the old sequential pipeline before it can duplicate Phase1"
tmux kill-session -t "${PACK_SESSION}" 2>/dev/null || true

while [[ "$(tracker_iteration)" != "${phase1_target}" ]]; do
  if ! tmux has-session -t "${PHASE1_SESSION}" 2>/dev/null; then
    log "ERROR Phase1 session exited at iteration $(tracker_iteration)"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

if tmux has-session -t "${PHASE23_SESSION}" 2>/dev/null; then
  log "Phase2/3 session already exists: ${PHASE23_SESSION}"
  exit 0
fi

phase23_command="exec bash ${REPO_ROOT}/scripts/run_qwen0p5b_unist198_all_phases.sh --config ${CONFIG_FILE} --start-phase phase2 --end-phase phase3"
tmux new-session -d -s "${PHASE23_SESSION}" -c "${REPO_ROOT}" "${phase23_command}"
log "Phase1 complete; started ${PHASE23_SESSION} for Phase2 and Phase3 training"
