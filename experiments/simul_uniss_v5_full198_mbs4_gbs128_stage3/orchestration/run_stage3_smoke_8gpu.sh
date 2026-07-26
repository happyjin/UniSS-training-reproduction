#!/usr/bin/env bash
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SMOKE_NAME="stage03_mbs4_gbs128_smoke_v1"
SMOKE_ROOT="${SIMUL_CHECKPOINT_ROOT}/${SMOKE_NAME}"
SMOKE_RUN_DIR="${RUN_DIR}/${SMOKE_NAME}"
SMOKE_LOG_DIR="${LOG_DIR}/${SMOKE_NAME}"
SMOKE_TENSORBOARD_DIR="${TENSORBOARD_DIR}/${SMOKE_NAME}"
MARKER="${SMOKE_RUN_DIR}/SMOKE_COMPLETE"
command="env LOG_DIR=${SMOKE_LOG_DIR} STAGE3_SAVE_ROOT=${SMOKE_ROOT} STAGE3_TENSORBOARD_DIR=${SMOKE_TENSORBOARD_DIR} STAGE3_TRAIN_ITERS=20 STAGE3_QWEN_WARMUP_ITERS=0 SIMUL_QWEN_SAVE_INTERVAL=20 SIMUL_QWEN_EVAL_INTERVAL=20 SIMUL_QWEN_EVAL_ITERS=1 SIMUL_FULL_VALIDATION=0 ${EXPERIMENT_DIR}/stage03_action_sft/run.sh"
if [[ "${DRY_RUN}" == "1" ]]; then bash -lc "${command} --dry-run"; exit 0; fi
[[ -f "${FULL_DATA_READY_MARKER}" && -f "${TRAINING_SCHEDULE_FILE}" ]] || exit 1
[[ ! -e "${SMOKE_ROOT}" && ! -e "${SMOKE_RUN_DIR}" && ! -e "${SMOKE_LOG_DIR}" ]] || {
  echo "Refusing existing MBS4 smoke output" >&2; exit 1;
}
mkdir -p "${SMOKE_RUN_DIR}" "${SMOKE_LOG_DIR}"
bash -lc "${command}"
[[ "$(tr -d '[:space:]' < "${SMOKE_ROOT}/latest_checkpointed_iteration.txt")" == "20" ]] || exit 1
log="${SMOKE_LOG_DIR}/stage_action_qwen.log"
! grep -Eq 'number of (skipped|nan) iterations: +[1-9]' "${log}"
printf 'completed_at=%s\nmicro_batch=%s\nglobal_batch=%s\n' \
  "$(date -u +%FT%TZ)" "${SIMUL_MICRO_BATCH_SIZE}" "${SIMUL_GLOBAL_BATCH_SIZE}" > "${MARKER}"
echo "Stage3 MBS4 smoke completed: ${MARKER}"
