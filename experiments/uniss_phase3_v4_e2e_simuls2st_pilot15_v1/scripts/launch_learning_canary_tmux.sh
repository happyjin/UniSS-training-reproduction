#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${LEARNING_RUN_ID:?set an immutable learning-canary run ID}"
SESSION=${SESSION:-${LEARNING_RUN_ID}}
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}

command=(env LEARNING_RUN_ID="${LEARNING_RUN_ID}")
for name in DATA_RUN_ID LEARNING_ITERS LEARNING_MBS LEARNING_GBS \
  LEARNING_NUM_WORKERS LEARNING_MASTER_PORT TASK_POOL_RUN_ID TEACHER_RUN_ID \
  STRUCTURAL_CANARY_RUN_ID; do
  if [[ -n "${!name:-}" ]]; then
    command+=("${name}=${!name}")
  fi
done
command+=("${SCRIPT_DIR}/run_learning_canary_8gpu.sh")
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" "cd $(printf %q "${REPO_ROOT}") && ${quoted}"

echo "started=${SESSION}"
echo "tensorboard=${TENSORBOARD_ROOT}/learning_canaries/${LEARNING_RUN_ID}"
echo "log=${LOG_ROOT}/learning_canaries/${LEARNING_RUN_ID}.log"
