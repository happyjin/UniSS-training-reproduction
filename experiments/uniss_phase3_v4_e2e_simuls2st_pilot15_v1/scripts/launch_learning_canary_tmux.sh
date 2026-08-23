#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${LEARNING_RUN_ID:?set an immutable learning-canary run ID}"
SESSION=${SESSION:-${LEARNING_RUN_ID}}
BOOTSTRAP_LOG=${LOG_ROOT}/learning_canaries/${LEARNING_RUN_ID}.bootstrap.log
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}
[[ ! -e "${BOOTSTRAP_LOG}" ]] || {
  echo "refusing to overwrite learning-canary bootstrap log: ${BOOTSTRAP_LOG}" >&2
  exit 2
}
mkdir -p "$(dirname -- "${BOOTSTRAP_LOG}")"

command=(env LEARNING_RUN_ID="${LEARNING_RUN_ID}")
for name in DATA_RUN_ID LEARNING_ITERS LEARNING_MBS LEARNING_GBS \
  LEARNING_NUM_WORKERS LEARNING_MASTER_PORT TASK_POOL_RUN_ID TEACHER_RUN_ID \
  STRUCTURAL_CANARY_RUN_ID LEARNING_PHASE_STRATIFIED \
  LEARNING_CONTENT_END_WEIGHT LEARNING_SEMANTIC_END_WEIGHT \
  LEARNING_SEMANTIC_END_MARGIN_WEIGHT LEARNING_SEMANTIC_END_LOGIT_MARGIN \
  LEARNING_SEMANTIC_PREFIX_CORRUPTION_RATE \
  LEARNING_SEMANTIC_PREFIX_CORRUPTION_TAIL \
  LEARNING_SEMANTIC_PREFIX_CORRUPTION_RAMP_UPDATES \
  LEARNING_SEMANTIC_BOUNDARY_ROLLIN_RATE \
  LEARNING_SEMANTIC_BOUNDARY_ROLLIN_RAMP_UPDATES; do
  if [[ -n "${!name:-}" ]]; then
    command+=("${name}=${!name}")
  fi
done
command+=("${SCRIPT_DIR}/run_learning_canary_8gpu.sh")
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" \
  "cd $(printf %q "${REPO_ROOT}") && ${quoted} > $(printf %q "${BOOTSTRAP_LOG}") 2>&1"

echo "started=${SESSION}"
echo "tensorboard=${TENSORBOARD_ROOT}/learning_canaries/${LEARNING_RUN_ID}"
echo "log=${LOG_ROOT}/learning_canaries/${LEARNING_RUN_ID}.log"
echo "bootstrap_log=${BOOTSTRAP_LOG}"
