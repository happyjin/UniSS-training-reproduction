#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
NAME="stage03_seq18000_mbs2_gbs128_smoke_v1"
ROOT="${SIMUL_CHECKPOINT_ROOT}/${NAME}"
SMOKE_RUN="${RUN_DIR}/${NAME}"
SMOKE_LOG="${LOG_DIR}/${NAME}"
MARKER="${SMOKE_RUN}/SMOKE_COMPLETE"
[[ -f "${FULL_DATA_READY_MARKER}" ]] || { echo "Data not ready" >&2; exit 1; }
[[ ! -e "${ROOT}" && ! -e "${SMOKE_RUN}" && ! -e "${SMOKE_LOG}" ]] || {
  echo "Refusing existing smoke output" >&2; exit 1;
}
mkdir -p "${SMOKE_RUN}" "${SMOKE_LOG}"
env LOG_DIR="${SMOKE_LOG}" STAGE3_SAVE_ROOT="${ROOT}" \
  STAGE3_TENSORBOARD_DIR="${TENSORBOARD_DIR}/${NAME}" STAGE3_TRAIN_ITERS=20 \
  STAGE3_QWEN_WARMUP_ITERS=0 SIMUL_QWEN_SAVE_INTERVAL=20 \
  SIMUL_QWEN_EVAL_INTERVAL=20 SIMUL_QWEN_EVAL_ITERS=1 SIMUL_FULL_VALIDATION=0 \
  "${EXPERIMENT_DIR}/stage03_action_sft/run.sh"
[[ "$(tr -d '[:space:]' < "${ROOT}/latest_checkpointed_iteration.txt")" == "20" ]] || exit 1
log="${SMOKE_LOG}/stage_action_qwen.log"
! grep -Eq 'number of (skipped|nan) iterations: +[1-9]' "${log}"
printf 'completed_at=%s\nseq_length=%s\nmicro_batch=%s\nglobal_batch=%s\n' \
  "$(date -u +%FT%TZ)" "${SEQ_LENGTH}" "${SIMUL_MICRO_BATCH_SIZE}" "${SIMUL_GLOBAL_BATCH_SIZE}" > "${MARKER}"
echo "Smoke complete: ${MARKER}"
