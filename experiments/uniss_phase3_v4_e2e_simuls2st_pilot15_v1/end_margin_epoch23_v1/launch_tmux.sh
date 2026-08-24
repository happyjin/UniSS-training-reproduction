#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"
SESSION=${SESSION:-endmargin_epoch23_train}
TB_SESSION=${TB_SESSION:-endmargin_epoch23_tb}
POST_SESSION=${POST_SESSION:-endmargin_epoch23_post}
TB_PORT=${TB_PORT:-6045}
BOOTSTRAP_LOG=${LOG_ROOT}/extended_canaries/${RUN_ID}.bootstrap.log
TB_LOG=${LOG_ROOT}/extended_canaries/${RUN_ID}.tensorboard.log
TB_DIR=${TENSORBOARD_ROOT}/extended_canaries/${RUN_ID}

for session in "${SESSION}" "${TB_SESSION}" "${POST_SESSION}"; do
  tmux has-session -t "${session}" 2>/dev/null && {
    echo "tmux session already exists: ${session}" >&2
    exit 2
  }
done
for path in "${BOOTSTRAP_LOG}" "${TB_LOG}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname -- "${BOOTSTRAP_LOG}")"

command=(env RUN_ID="${RUN_ID}" DATA_RUN_ID="${DATA_RUN_ID}")
for name in PARENT_RUN_ID TASK_POOL_RUN_ID TEACHER_RUN_ID \
  STRUCTURAL_CANARY_RUN_ID MASTER_PORT RUN_SEED; do
  [[ -n "${!name:-}" ]] && command+=("${name}=${!name}")
done
command+=("${HERE}/run_8gpu.sh")
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" \
  "cd $(printf %q "${REPO_ROOT}") && ${quoted} > $(printf %q "${BOOTSTRAP_LOG}") 2>&1"

tb_command="while [[ ! -d $(printf %q "${TB_DIR}") ]]; do sleep 2; done; exec $(printf %q "${PYTHON_BIN}") -m tensorboard.main --logdir $(printf %q "${TB_DIR}") --host 0.0.0.0 --port $(printf %q "${TB_PORT}") --load_fast=false >> $(printf %q "${TB_LOG}") 2>&1"
tmux new-session -d -s "${TB_SESSION}" "${tb_command}"

post_command=(env TRAIN_RUN_ID="${RUN_ID}" TRAIN_SESSION="${SESSION}")
post_command+=("${HERE}/wait_then_export_gates.sh")
printf -v post_quoted '%q ' "${post_command[@]}"
tmux new-session -d -s "${POST_SESSION}" \
  "cd $(printf %q "${REPO_ROOT}") && ${post_quoted}"

echo "started=${SESSION}"
echo "post_session=${POST_SESSION}"
echo "tensorboard_session=${TB_SESSION}"
echo "tensorboard=http://127.0.0.1:${TB_PORT}"
echo "bootstrap_log=${BOOTSTRAP_LOG}"
echo "training_log=${LOG_ROOT}/extended_canaries/${RUN_ID}.log"
