#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit4/config.env"
export USER_ROOT REPO_ROOT ENV_ROOT PYTHON EXPERIMENT_NAME
export PHASE3_NATIVE_CHECKPOINT PHASE3_REPLAY_PACKED PHASE3_REPLAY_OFFSETS
export VALID_REPLAY_PACKED VALID_REPLAY_OFFSETS TRAJECTORY_PACKED
export VALID_TRAJECTORY_PACKED TRAINING_ROOT TRAINING_MANIFEST REPLAY_SUBSET_OFFSETS
export SAVE_DIR RUN_DIR TB_DIR LOG_PATH TENSORBOARD_PORT CUDA_VISIBLE_DEVICES
export MASTER_PORT MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LENGTH COVERAGE_EPOCHS
export REPLAY_FRACTION LR_QWEN_LORA LR_FRONTEND LR_NEW_HEADS MIN_LR
export RUN_ENTRYPOINT="${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit4/pretrain_overfit4.py"
exec bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/run_8gpu.sh" "$@"
