#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SMOKE_NAME="shuffle_smoke_8gpu_gbs128_v1"
SMOKE_ROOT="${SIMUL_CHECKPOINT_ROOT}/${SMOKE_NAME}"
SMOKE_RUN_DIR="${RUN_DIR}/${SMOKE_NAME}"
SMOKE_LOG_DIR="${LOG_DIR}/${SMOKE_NAME}"
SMOKE_TENSORBOARD_DIR="${TENSORBOARD_DIR}/${SMOKE_NAME}"
COMPLETE_MARKER="${SMOKE_RUN_DIR}/SHUFFLE_SMOKE_COMPLETE"
stage03="${SMOKE_ROOT}/stage03_action_sft"
stage04="${SMOKE_ROOT}/stage04_interleaved_s2st"
stage06="${SMOKE_ROOT}/stage06_joint_refinement"
commands=(
  "env LOG_DIR=${SMOKE_LOG_DIR} STAGE3_SAVE_ROOT=${stage03} STAGE3_TENSORBOARD_DIR=${SMOKE_TENSORBOARD_DIR}/stage03 STAGE3_TRAIN_ITERS=2 STAGE3_QWEN_WARMUP_ITERS=0 SIMUL_QWEN_SAVE_INTERVAL=2 SIMUL_QWEN_EVAL_INTERVAL=2 ${EXPERIMENT_DIR}/stage03_action_sft/run.sh"
  "env LOG_DIR=${SMOKE_LOG_DIR} STAGE4_LOAD_ROOT=${stage03} STAGE4_SAVE_ROOT=${stage04} STAGE4_TENSORBOARD_DIR=${SMOKE_TENSORBOARD_DIR}/stage04 STAGE4_TRAIN_ITERS=2 STAGE4_QWEN_WARMUP_ITERS=0 SIMUL_QWEN_SAVE_INTERVAL=2 SIMUL_QWEN_EVAL_INTERVAL=2 ${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh"
  "env LOG_DIR=${SMOKE_LOG_DIR} STAGE6_LOAD_ROOT=${stage04} STAGE6_SAVE_ROOT=${stage06} STAGE6_TENSORBOARD_DIR=${SMOKE_TENSORBOARD_DIR}/stage06 STAGE6_TRAIN_ITERS=2 STAGE6_QWEN_WARMUP_ITERS=0 SIMUL_QWEN_SAVE_INTERVAL=2 SIMUL_QWEN_EVAL_INTERVAL=2 ${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  for command in "${commands[@]}"; do bash -lc "${command} --dry-run"; done
  exit 0
fi
[[ -f "${FULL_DATA_READY_MARKER}" && -f "${TRAINING_SCHEDULE_FILE}" ]] || {
  echo "Full data or GBS128 schedule is not ready" >&2; exit 1;
}
[[ ! -e "${SMOKE_ROOT}" && ! -e "${SMOKE_RUN_DIR}" && ! -e "${SMOKE_LOG_DIR}" ]] || {
  echo "Refusing to overwrite existing GBS128 smoke output" >&2; exit 1;
}
mkdir -p "${SMOKE_RUN_DIR}" "${SMOKE_LOG_DIR}"
for command in "${commands[@]}"; do bash -lc "${command}"; done
for root in "${stage03}" "${stage04}" "${stage06}"; do
  [[ "$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")" == "2" ]] || exit 1
done
printf 'completed_at=%s\nmicro_batch=%s\nglobal_batch=%s\n' \
  "$(date -u +%FT%TZ)" "${SIMUL_MICRO_BATCH_SIZE}" "${SIMUL_GLOBAL_BATCH_SIZE}" > "${COMPLETE_MARKER}"
echo "Eight-GPU GBS128 shuffle smoke completed: ${COMPLETE_MARKER}"
