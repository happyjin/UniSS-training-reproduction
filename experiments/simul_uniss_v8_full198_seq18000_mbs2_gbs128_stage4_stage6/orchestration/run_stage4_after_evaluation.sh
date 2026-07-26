#!/usr/bin/env bash
set -euo pipefail

EVALUATION_ROOT=""
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --evaluation-root) EVALUATION_ROOT="$2"; shift 2 ;;
    --wait-interval) WAIT_INTERVAL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${EVALUATION_ROOT}" ]] || { echo "--evaluation-root is required" >&2; exit 2; }
[[ "${WAIT_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "--wait-interval must be positive" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
EVALUATION_ROOT="$(realpath -m "${EVALUATION_ROOT}")"
STAGE4_LOG="${LOG_DIR}/stage_interleaved_qwen.log"
HANDOFF_MARKER="${RUN_DIR}/STAGE4_AFTER_EVALUATION_COMPLETE"

verify_stage4() {
  local pointer="${STAGE4_SAVE_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${pointer}" ]] || return 1
  [[ "$(tr -d '[:space:]' < "${pointer}")" == "${STAGE4_TRAIN_ITERS}" ]] || return 1
  local iteration_dir
  printf -v iteration_dir '%s/iter_%07d' "${STAGE4_SAVE_ROOT}" "${STAGE4_TRAIN_ITERS}"
  [[ -d "${iteration_dir}" ]] || return 1
  [[ "$(find "${iteration_dir}" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -ge 8 ]] || return 1
  [[ -f "${STAGE4_LOG}" ]] || return 1
  grep -Eq "iteration +${STAGE4_TRAIN_ITERS}/ +${STAGE4_TRAIN_ITERS}" "${STAGE4_LOG}" || return 1
  grep -Eq "validation loss at iteration +${STAGE4_TRAIN_ITERS}" "${STAGE4_LOG}" || return 1
  ! grep -Eq 'number of (skipped|nan) iterations: +[1-9]' "${STAGE4_LOG}"
}

wait_for_file() {
  local path="$1" label="$2"
  while [[ ! -f "${path}" ]]; do
    printf '%s waiting_for_%s path=%s\n' "$(date -u +%FT%TZ)" "${label}" "${path}"
    sleep "${WAIT_INTERVAL}"
  done
}

wait_for_idle_gpus() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    printf '%s waiting_for_idle_gpus\n' "$(date -u +%FT%TZ)"
    sleep "${WAIT_INTERVAL}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "evaluation_root=${EVALUATION_ROOT}"
  echo "evaluation_complete=${EVALUATION_ROOT}/COMPLETE"
  echo "evaluation_report=${EVALUATION_ROOT}/report/phase2_phase3_detailed_evaluation_report.md"
  echo "data_ready=${FULL_DATA_READY_MARKER}"
  echo "seq_length=${SEQ_LENGTH} micro_batch=${SIMUL_MICRO_BATCH_SIZE} global_batch=${SIMUL_GLOBAL_BATCH_SIZE}"
  STAGE4_TRAIN_ITERS="${STAGE4_TRAIN_ITERS:-100}" STAGE4_QWEN_WARMUP_ITERS="${STAGE4_QWEN_WARMUP_ITERS:-5}" \
    "${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh" --dry-run
  exit 0
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
[[ ! -e "${HANDOFF_MARKER}" ]] || { echo "Stage4 handoff already complete: ${HANDOFF_MARKER}"; exit 0; }

wait_for_file "${EVALUATION_ROOT}/COMPLETE" evaluation_complete
wait_for_file "${EVALUATION_ROOT}/report/phase2_phase3_detailed_evaluation_report.md" detailed_report
wait_for_file "${FULL_DATA_READY_MARKER}" interleaved_18k_data
wait_for_file "${V7_STAGE3_ROOT}/latest_checkpointed_iteration.txt" v7_stage3_checkpoint
[[ "$(tr -d '[:space:]' < "${V7_STAGE3_ROOT}/latest_checkpointed_iteration.txt")" == "${V7_STAGE3_REQUIRED_ITERATION}" ]] || {
  echo "v7 Stage3 did not finish at required iteration ${V7_STAGE3_REQUIRED_ITERATION}" >&2
  exit 1
}

if verify_stage4; then
  echo "Verified Stage4 is already complete; preserving existing output."
else
  [[ ! -e "${STAGE4_SAVE_ROOT}" && ! -e "${STAGE4_TENSORBOARD_DIR}" && ! -e "${STAGE4_LOG}" ]] || {
    echo "Refusing partial or unverifiable Stage4 output" >&2
    exit 1
  }
  wait_for_idle_gpus
  if ! tmux has-session -t "${TENSORBOARD_SESSION}" 2>/dev/null; then
    tb_command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/orchestration/start_tensorboard.sh")"
    tmux new-session -d -s "${TENSORBOARD_SESSION}" "bash -lc $(printf '%q' "${tb_command}")"
  fi
  "${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh"
  verify_stage4 || { echo "Stage4 completion verification failed" >&2; exit 1; }
fi

printf '%s\n' \
  "completed_at=$(date -u +%FT%TZ)" \
  "evaluation_root=${EVALUATION_ROOT}" \
  "evaluation_report=${EVALUATION_ROOT}/report/phase2_phase3_detailed_evaluation_report.md" \
  "stage=stage04_interleaved_s2st" \
  "seq_length=${SEQ_LENGTH}" \
  "micro_batch=${SIMUL_MICRO_BATCH_SIZE}" \
  "global_batch=${SIMUL_GLOBAL_BATCH_SIZE}" \
  "checkpoint_root=${STAGE4_SAVE_ROOT}" \
  >"${HANDOFF_MARKER}"
echo "Stage4 after-evaluation handoff complete: ${HANDOFF_MARKER}"
