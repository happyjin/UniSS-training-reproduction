#!/usr/bin/env bash
set -euo pipefail
RECOVER_COMPLETED=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recover-completed) RECOVER_COMPLETED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"

verify_stage() {
  local label="$1" root="$2" expected="$3" log="$4"
  local pointer="${root}/latest_checkpointed_iteration.txt"
  [[ -f "${pointer}" ]] || return 1
  [[ "$(tr -d '[:space:]' < "${pointer}")" == "${expected}" ]] || return 1
  local iteration_dir
  printf -v iteration_dir '%s/iter_%07d' "${root}" "${expected}"
  [[ -d "${iteration_dir}" ]] || return 1
  [[ "$(find "${iteration_dir}" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -ge 8 ]] || return 1
  [[ -f "${log}" ]] || return 1
  if ! grep -Eq "iteration +${expected}/ +${expected}" "${log}"; then
    grep -Eq "successfully saved checkpoint from iteration +${expected}" "${log}" || return 1
  fi
  grep -Eq "validation loss at iteration +${expected}" "${log}" || return 1
  ! grep -Eq 'number of (skipped|nan) iterations: +[1-9]' "${log}"
}

require_v7_stage3_complete() {
  local pointer="${V7_STAGE3_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${pointer}" ]] || { echo "v7 Stage3 checkpoint is not saved yet" >&2; return 1; }
  [[ "$(tr -d '[:space:]' < "${pointer}")" == "${V7_STAGE3_REQUIRED_ITERATION}" ]] || {
    echo "v7 Stage3 has not reached required iteration ${V7_STAGE3_REQUIRED_ITERATION}" >&2
    return 1
  }
}

if [[ "${DRY_RUN}" == "1" ]]; then
  STAGE4_TRAIN_ITERS="${STAGE4_TRAIN_ITERS:-100}" STAGE4_QWEN_WARMUP_ITERS="${STAGE4_QWEN_WARMUP_ITERS:-5}" \
    "${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh" --dry-run
  STAGE6_TRAIN_ITERS="${STAGE6_TRAIN_ITERS:-25}" STAGE6_QWEN_WARMUP_ITERS="${STAGE6_QWEN_WARMUP_ITERS:-2}" \
    "${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh" --dry-run
  exit 0
fi

[[ -f "${FULL_DATA_READY_MARKER}" ]] || { echo "18k interleaved data is not ready" >&2; exit 1; }
require_v7_stage3_complete
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

stage4_log="${LOG_DIR}/stage_interleaved_qwen.log"
stage6_log="${LOG_DIR}/stage_joint_qwen.log"
pipeline_marker="${RUN_DIR}/STAGE4_STAGE6_PIPELINE_COMPLETE"
[[ ! -e "${pipeline_marker}" ]] || { echo "Pipeline is already complete: ${pipeline_marker}"; exit 0; }

if verify_stage stage04 "${STAGE4_SAVE_ROOT}" "${STAGE4_TRAIN_ITERS}" "${stage4_log}"; then
  [[ "${RECOVER_COMPLETED}" == "1" ]] || { echo "Stage4 exists; pass --recover-completed" >&2; exit 1; }
  echo "Recovered verified completed Stage4"
else
  [[ ! -e "${STAGE4_SAVE_ROOT}" && ! -e "${STAGE4_TENSORBOARD_DIR}" && ! -e "${stage4_log}" ]] || {
    echo "Refusing partial or unverifiable Stage4 output" >&2; exit 1;
  }
  "${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh"
  verify_stage stage04 "${STAGE4_SAVE_ROOT}" "${STAGE4_TRAIN_ITERS}" "${stage4_log}" || {
    echo "Stage4 completion verification failed" >&2; exit 1;
  }
fi

if verify_stage stage06 "${STAGE6_SAVE_ROOT}" "${STAGE6_TRAIN_ITERS}" "${stage6_log}"; then
  [[ "${RECOVER_COMPLETED}" == "1" ]] || { echo "Stage6 exists; pass --recover-completed" >&2; exit 1; }
  echo "Recovered verified completed Stage6"
else
  [[ ! -e "${STAGE6_SAVE_ROOT}" && ! -e "${STAGE6_TENSORBOARD_DIR}" && ! -e "${stage6_log}" ]] || {
    echo "Refusing partial or unverifiable Stage6 output" >&2; exit 1;
  }
  "${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh"
  verify_stage stage06 "${STAGE6_SAVE_ROOT}" "${STAGE6_TRAIN_ITERS}" "${stage6_log}" || {
    echo "Stage6 completion verification failed" >&2; exit 1;
  }
fi

printf 'completed_at=%s\nseq_length=%s\nmicro_batch=%s\nglobal_batch=%s\n' \
  "$(date -u +%FT%TZ)" "${SEQ_LENGTH}" "${SIMUL_MICRO_BATCH_SIZE}" "${SIMUL_GLOBAL_BATCH_SIZE}" > "${pipeline_marker}"
echo "Stage4/Stage6 pipeline complete: ${pipeline_marker}"
