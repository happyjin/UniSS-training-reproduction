#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

SMOKE_NAME="${SHUFFLE_SMOKE_NAME:-shuffle_smoke_8gpu_v2}"
SMOKE_ROOT="${SIMUL_CHECKPOINT_ROOT}/${SMOKE_NAME}"
SMOKE_RUN_DIR="${RUN_DIR}/${SMOKE_NAME}"
SMOKE_LOG_DIR="${LOG_DIR}/${SMOKE_NAME}"
SMOKE_TENSORBOARD_DIR="${TENSORBOARD_DIR}/${SMOKE_NAME}"
COMPLETE_MARKER="${SMOKE_RUN_DIR}/SHUFFLE_SMOKE_COMPLETE"

stage03_root="${SMOKE_ROOT}/stage03_action_sft"
stage04_root="${SMOKE_ROOT}/stage04_interleaved_s2st"
stage06_root="${SMOKE_ROOT}/stage06_joint_refinement"

stage03_cmd=(env
  LOG_DIR="${SMOKE_LOG_DIR}"
  STAGE3_SAVE_ROOT="${stage03_root}"
  STAGE3_TENSORBOARD_DIR="${SMOKE_TENSORBOARD_DIR}/stage03_action_sft"
  STAGE3_TRAIN_ITERS=2
  SIMUL_QWEN_WARMUP_ITERS=0
  SIMUL_QWEN_SAVE_INTERVAL=2
  SIMUL_QWEN_EVAL_INTERVAL=2
  "${EXPERIMENT_DIR}/stage03_action_sft/run.sh")
stage04_cmd=(env
  LOG_DIR="${SMOKE_LOG_DIR}"
  STAGE4_LOAD_ROOT="${stage03_root}"
  STAGE4_SAVE_ROOT="${stage04_root}"
  STAGE4_TENSORBOARD_DIR="${SMOKE_TENSORBOARD_DIR}/stage04_interleaved_s2st"
  STAGE4_TRAIN_ITERS=2
  SIMUL_QWEN_WARMUP_ITERS=0
  SIMUL_QWEN_SAVE_INTERVAL=2
  SIMUL_QWEN_EVAL_INTERVAL=2
  "${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh")
stage06_cmd=(env
  LOG_DIR="${SMOKE_LOG_DIR}"
  STAGE6_LOAD_ROOT="${stage04_root}"
  STAGE6_SAVE_ROOT="${stage06_root}"
  STAGE6_TENSORBOARD_DIR="${SMOKE_TENSORBOARD_DIR}/stage06_joint_refinement"
  STAGE6_TRAIN_ITERS=2
  SIMUL_QWEN_WARMUP_ITERS=0
  SIMUL_QWEN_SAVE_INTERVAL=2
  SIMUL_QWEN_EVAL_INTERVAL=2
  "${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh")

if [[ "${DRY_RUN}" == "1" ]]; then
  "${stage03_cmd[@]}" --dry-run
  "${stage04_cmd[@]}" --dry-run
  "${stage06_cmd[@]}" --dry-run
  exit 0
fi

if [[ -e "${SMOKE_ROOT}" || -e "${SMOKE_RUN_DIR}" || -e "${SMOKE_LOG_DIR}" ]]; then
  echo "Refusing to overwrite an existing shuffle smoke directory." >&2
  echo "checkpoint=${SMOKE_ROOT}" >&2
  echo "run=${SMOKE_RUN_DIR}" >&2
  echo "log=${SMOKE_LOG_DIR}" >&2
  exit 1
fi

mkdir -p "${SMOKE_RUN_DIR}" "${SMOKE_LOG_DIR}"

verify_stage() {
  local stage="$1" root="$2" log_file="$3"
  local iteration
  [[ -f "${root}/latest_checkpointed_iteration.txt" ]] || {
    echo "${stage}: missing checkpoint pointer" >&2
    return 1
  }
  iteration="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")"
  [[ "${iteration}" == "2" ]] || {
    echo "${stage}: expected iteration 2, got ${iteration}" >&2
    return 1
  }
  rg -q 'data_sharding[. ]+False' "${log_file}"
  rg -q 'dataloader_type[. ]+cyclic' "${log_file}"
  rg -q 'full_validation[. ]+True' "${log_file}"
  rg -q 'seed[. ]+20260725' "${log_file}"
  rg -q 'number of nan iterations:[ ]+0' "${log_file}"
}

"${stage03_cmd[@]}"
verify_stage stage03 "${stage03_root}" "${SMOKE_LOG_DIR}/stage_action_qwen.log"
"${stage04_cmd[@]}"
verify_stage stage04 "${stage04_root}" "${SMOKE_LOG_DIR}/stage_interleaved_qwen.log"
"${stage06_cmd[@]}"
verify_stage stage06 "${stage06_root}" "${SMOKE_LOG_DIR}/stage_joint_qwen.log"

printf 'completed_at=%s\nanchor=%s\ngpus=%s\n' \
  "$(date -u +%FT%TZ)" "${QWEN_CHECKPOINT_ROOT}" "${SIMUL_CUDA_VISIBLE_DEVICES}" \
  > "${COMPLETE_MARKER}"
echo "Eight-GPU global-shuffle smoke completed: ${COMPLETE_MARKER}"
