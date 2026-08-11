#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRAIN_SESSION="uniss_dense_aligned_pilot15_train"
TB_SESSION="uniss_dense_aligned_pilot15_tb"

if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  echo "Training tmux already exists: ${TRAIN_SESSION}" >&2
  exit 1
fi
if ! tmux has-session -t "${TB_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${TB_SESSION}" \
    "cd '${REPO_ROOT}' && bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/start_tensorboard.sh 2>&1 | tee -a logs/uniss_phase3_dense_aligned_streaming_pilot15_v1_tensorboard.log"
fi
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd '${REPO_ROOT}' && bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_8gpu.sh 2>&1 | tee -a logs/uniss_phase3_dense_aligned_streaming_pilot15_v1_launcher.log"
echo "training_tmux=${TRAIN_SESSION}"
echo "tensorboard_tmux=${TB_SESSION}"
