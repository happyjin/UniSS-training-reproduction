#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/uniss_phase2}"
HF_REFERENCE="${HF_REFERENCE:-${REPO_ROOT}/checkpoints/qwen2_1p5b_uniss_vocab_hf}"
HF_EXPORT_ROOT="${HF_EXPORT_ROOT:-${REPO_ROOT}/checkpoints/exported_hf}"
RUN_NAME="${RUN_NAME:-$(basename "${CHECKPOINT_DIR}")}"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/eval_outputs/.watch_state/${RUN_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
ONCE="${ONCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SPLIT="${SPLIT:-dev}"
LIMIT_RECORDS="${LIMIT_RECORDS:-8}"
MODES="${MODES:-quality performance}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "${HF_EXPORT_ROOT}" "${STATE_DIR}" "${REPO_ROOT}/eval_outputs"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

iter_number() {
  local name="$1"
  name="${name##*/}"
  name="${name#iter_}"
  printf '%d\n' "${name#0}"
}

is_stable_dir() {
  local path="$1"
  local newest
  newest="$(find "${path}" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1)"
  if [[ -z "${newest}" ]]; then
    newest="$(stat -c '%Y' "${path}")"
  fi
  python - "$newest" "$STABILITY_SECONDS" <<'PY'
import sys, time
newest = float(sys.argv[1])
stability = float(sys.argv[2])
raise SystemExit(0 if time.time() - newest >= stability else 1)
PY
}

process_checkpoint() {
  local iter_dir="$1"
  local iter_name
  iter_name="$(basename "${iter_dir}")"
  local marker="${STATE_DIR}/${iter_name}.done"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if ! is_stable_dir "${iter_dir}"; then
    log "skip unstable checkpoint ${iter_name}"
    return 0
  fi

  local hf_output="${HF_EXPORT_ROOT}/${RUN_NAME}_${iter_name}"
  log "export ${iter_name} -> ${hf_output}"
  run_cmd "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
    --hf-model "${HF_REFERENCE}" \
    --megatron-path "${iter_dir}" \
    --hf-output "${hf_output}"

  log "generate ${SPLIT} audio for ${iter_name}"
  run_cmd env \
    HF_CHECKPOINT="${hf_output}" \
    STEP_NAME="${RUN_NAME}_${iter_name}" \
    SPLIT="${SPLIT}" \
    LIMIT_RECORDS="${LIMIT_RECORDS}" \
    MODES="${MODES}" \
    EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    "${REPO_ROOT}/scripts/generate_unist_audio_eval.sh"

  if [[ "${DRY_RUN}" != "1" ]]; then
    touch "${marker}"
  fi
}

main_loop() {
  while true; do
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
      log "waiting for checkpoint dir ${CHECKPOINT_DIR}"
    else
      mapfile -t iter_dirs < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'iter_*' | sort -V)
      for iter_dir in "${iter_dirs[@]}"; do
        process_checkpoint "${iter_dir}"
        if [[ "${ONCE}" == "1" ]]; then
          return 0
        fi
      done
    fi

    if [[ "${ONCE}" == "1" ]]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

main_loop
