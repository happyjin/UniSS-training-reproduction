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

PIPELINE_SESSION="${PIPELINE_SESSION:-unist198_phase23_train}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_PHASE3_ITERATION="${MIN_PHASE3_ITERATION:-10}"
MONITOR_LOG="${MONITOR_LOG:-${REPO_ROOT}/logs/uniss_qwen0p5b_phase2_phase3_health.log}"
HEALTHY_START_MARKER="${HEALTHY_START_MARKER:-${RUN_DIR}/PHASE3_HEALTHY_START}"
MONITOR_COMPLETE_MARKER="${MONITOR_COMPLETE_MARKER:-${RUN_DIR}/PHASE3_MONITOR_COMPLETE}"
MONITOR_FAILURE_MARKER="${MONITOR_FAILURE_MARKER:-${RUN_DIR}/PHASE2_PHASE3_MONITOR_FAILED}"
PHASE2_TRACKER="${PHASE2_SAVE}/latest_checkpointed_iteration.txt"
PHASE3_TRACKER="${PHASE3_SAVE}/latest_checkpointed_iteration.txt"

ceil_div() {
  echo $(( ($1 + $2 - 1) / $2 ))
}

read_count() {
  local phase="$1"
  local packed="$2"
  local override_name="${phase^^}_PACKED_COUNT_OVERRIDE"
  local override="${!override_name:-}"
  if [[ -n "${override}" ]]; then
    [[ "${override}" =~ ^[1-9][0-9]*$ ]] || {
      echo "Invalid ${override_name}: ${override}" >&2
      exit 1
    }
    echo "${override}"
    return
  fi
  local count_file="${packed}.count"
  [[ -s "${count_file}" ]] || { echo "Missing packed count: ${count_file}" >&2; exit 1; }
  local value
  value="$(<"${count_file}")"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid packed count in ${count_file}: ${value}" >&2
    exit 1
  }
  echo "${value}"
}

PHASE2_COUNT="$(read_count phase2 "${PHASE2_TRAIN}")"
PHASE3_COUNT="$(read_count phase3 "${PHASE3_TRAIN}")"
PHASE2_TARGET="${PHASE2_TRAIN_ITERS:-$(ceil_div "${PHASE2_COUNT}" "${GLOBAL_BATCH_SIZE}")}"
PHASE3_TARGET="${PHASE3_TRAIN_ITERS:-$(ceil_div "${PHASE3_COUNT}" "${GLOBAL_BATCH_SIZE}")}"

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_SECONDS must be positive" >&2; exit 1; }
[[ "${MIN_PHASE3_ITERATION}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MIN_PHASE3_ITERATION must be positive" >&2
  exit 1
}
if (( MIN_PHASE3_ITERATION > PHASE3_TARGET )); then
  echo "MIN_PHASE3_ITERATION=${MIN_PHASE3_ITERATION} exceeds Phase3 target ${PHASE3_TARGET}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] monitor ${PIPELINE_SESSION} without starting, stopping, or modifying training"
  echo "[dry-run] Phase2 target=${PHASE2_TARGET}; require final tracker before accepting transition"
  echo "[dry-run] Phase3 target=${PHASE3_TARGET}; require iteration>=${MIN_PHASE3_ITERATION}, Phase2 load, finite TensorBoard lm loss, and zero NaN/skips"
  echo "[dry-run] healthy marker=${HEALTHY_START_MARKER}"
  echo "[dry-run] completion marker=${MONITOR_COMPLETE_MARKER}"
  exit 0
fi

mkdir -p "$(dirname "${MONITOR_LOG}")" "${RUN_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${MONITOR_LOG}"
}

write_marker() {
  local marker="$1"
  shift
  local tmp="${marker}.tmp.$$"
  {
    printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
    printf '%s\n' "$@"
  } > "${tmp}"
  mv "${tmp}" "${marker}"
}

safe_tracker_iteration() {
  local tracker="$1"
  if [[ ! -s "${tracker}" ]]; then
    echo -1
    return
  fi
  local value
  value="$(tr -d '[:space:]' < "${tracker}")"
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value}"
  else
    echo invalid
  fi
}

fail_monitor() {
  local message="$1"
  log "ERROR ${message}"
  write_marker "${MONITOR_FAILURE_MARKER}" \
    "message=${message}" \
    "phase2_iteration=$(safe_tracker_iteration "${PHASE2_TRACKER}")" \
    "phase3_iteration=$(safe_tracker_iteration "${PHASE3_TRACKER}")"
  exit 1
}

tracker_iteration() {
  local tracker="$1"
  if [[ ! -s "${tracker}" ]]; then
    echo -1
    return
  fi
  local value
  value="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail_monitor "invalid checkpoint tracker ${tracker}: ${value}"
  echo "${value}"
}

latest_logged_iteration() {
  local log_path="$1"
  local expected_target="$2"
  if [[ ! -s "${log_path}" ]]; then
    echo -1
    return
  fi
  local line
  line="$(rg 'iteration +[0-9]+/ +[0-9]+' "${log_path}" 2>/dev/null | tail -n 1 || true)"
  if [[ "${line}" =~ iteration[[:space:]]+([0-9]+)/[[:space:]]+([0-9]+) ]]; then
    if [[ "${BASH_REMATCH[2]}" != "${expected_target}" ]]; then
      fail_monitor "log target ${BASH_REMATCH[2]} does not match expected ${expected_target}: ${log_path}"
    fi
    echo "${BASH_REMATCH[1]}"
  else
    echo -1
  fi
}

has_training_error() {
  local log_path="$1"
  [[ -s "${log_path}" ]] || return 1
  rg -q \
    'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|NCCL[^\n]*(error|Error)|number of skipped iterations: +[1-9]|number of nan iterations: +[1-9]' \
    "${log_path}"
}

pipeline_is_alive() {
  tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null || return 1
  pgrep -af 'scripts/run_qwen0p5b_unist198_all_phases.sh' 2>/dev/null \
    | rg -q -- '--start-phase phase2.*--end-phase phase3'
}

phase3_load_is_confirmed() {
  if rg -F -q "${PHASE2_SAVE}" "${PHASE3_LOG}" 2>/dev/null; then
    return 0
  fi
  pgrep -af 'training/pretrain_uniss_megatron.py' 2>/dev/null \
    | rg -F -- "--uniss-packed-train ${PHASE3_TRAIN}" \
    | rg -F -q -- "--load ${PHASE2_SAVE}"
}

phase3_tensorboard_is_healthy() {
  local python_bin="${ENV_ROOT}/bin/python"
  [[ -x "${python_bin}" ]] || return 1
  "${python_bin}" - "${TENSORBOARD_ROOT}/phase3" "${MIN_PHASE3_ITERATION}" >/dev/null 2>&1 <<'PY'
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sys.argv[1]
minimum_step = int(sys.argv[2])
events = EventAccumulator(path)
events.Reload()
scalars = events.Tags().get("scalars", [])
if "lm loss" not in scalars:
    raise SystemExit(1)
values = events.Scalars("lm loss")
if not values:
    raise SystemExit(1)
latest = values[-1]
raise SystemExit(0 if latest.step >= minimum_step and math.isfinite(latest.value) else 1)
PY
}

monitor_checkpoint_until() {
  local phase="$1"
  local tracker="$2"
  local log_path="$3"
  local target="$4"
  local last_reported=-2
  while true; do
    local checkpoint_iteration logged_iteration
    checkpoint_iteration="$(tracker_iteration "${tracker}")"
    logged_iteration="$(latest_logged_iteration "${log_path}" "${target}")"
    if [[ "${checkpoint_iteration}" != "${last_reported}" ]]; then
      log "${phase} progress: checkpoint=${checkpoint_iteration}/${target}, log_iteration=${logged_iteration}/${target}"
      last_reported="${checkpoint_iteration}"
    fi
    if (( checkpoint_iteration == target )); then
      return 0
    fi
    if (( checkpoint_iteration > target )); then
      fail_monitor "${phase} checkpoint ${checkpoint_iteration} exceeds target ${target}"
    fi
    if has_training_error "${log_path}"; then
      fail_monitor "${phase} log contains a fatal error, nonzero skipped iteration, or NaN iteration"
    fi
    if ! pipeline_is_alive; then
      fail_monitor "pipeline tmux session ${PIPELINE_SESSION} exited before ${phase} completed"
    fi
    sleep "${POLL_SECONDS}"
  done
}

rm -f "${MONITOR_FAILURE_MARKER}"
log "monitor started: session=${PIPELINE_SESSION}, Phase2=${PHASE2_TARGET}, Phase3=${PHASE3_TARGET}"

monitor_checkpoint_until phase2 "${PHASE2_TRACKER}" "${PHASE2_LOG}" "${PHASE2_TARGET}"
log "Phase2 final checkpoint confirmed at iteration ${PHASE2_TARGET}; waiting for Phase3 healthy start"

if [[ ! -f "${HEALTHY_START_MARKER}" ]]; then
  while true; do
    if has_training_error "${PHASE3_LOG}"; then
      fail_monitor "Phase3 log contains a fatal error, nonzero skipped iteration, or NaN iteration"
    fi
    phase3_logged_iteration="$(latest_logged_iteration "${PHASE3_LOG}" "${PHASE3_TARGET}")"
    if (( phase3_logged_iteration >= MIN_PHASE3_ITERATION )) \
        && phase3_load_is_confirmed \
        && phase3_tensorboard_is_healthy; then
      write_marker "${HEALTHY_START_MARKER}" \
        "phase2_iteration=${PHASE2_TARGET}" \
        "phase3_log_iteration=${phase3_logged_iteration}" \
        "phase3_target=${PHASE3_TARGET}" \
        "phase3_load=${PHASE2_SAVE}" \
        "tensorboard=${TENSORBOARD_ROOT}/phase3" \
        "nan_or_skipped=0"
      log "Phase3 healthy start confirmed at log iteration ${phase3_logged_iteration}; TensorBoard lm loss is finite"
      break
    fi
    if ! pipeline_is_alive; then
      fail_monitor "pipeline tmux session ${PIPELINE_SESSION} exited before Phase3 reached a healthy start"
    fi
    sleep "${POLL_SECONDS}"
  done
else
  log "existing Phase3 healthy-start marker found: ${HEALTHY_START_MARKER}"
fi

monitor_checkpoint_until phase3 "${PHASE3_TRACKER}" "${PHASE3_LOG}" "${PHASE3_TARGET}"

for _ in $(seq 1 10); do
  if [[ -f "${RUN_DIR}/TRAINING_COMPLETE" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done
[[ -f "${RUN_DIR}/TRAINING_COMPLETE" ]] || fail_monitor "Phase3 checkpoint is final but TRAINING_COMPLETE was not written"

write_marker "${MONITOR_COMPLETE_MARKER}" \
  "phase2_iteration=${PHASE2_TARGET}" \
  "phase3_iteration=${PHASE3_TARGET}" \
  "training_complete=${RUN_DIR}/TRAINING_COMPLETE" \
  "nan_or_skipped=0"
log "Phase2 -> Phase3 pipeline completed and final markers were verified"
