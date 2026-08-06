#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

export RUN_NAME="${RUN_NAME:-phase3_whisper_streamspeech_joint_smoke_v1}"
export TRAIN_MANIFEST="${SMOKE_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${SMOKE_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${SMOKE_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${SMOKE_ROOT}/direction_indices"
export REPLAY_OFFSETS="${SMOKE_REPLAY_OFFSETS}"
export ALLOW_PARTIAL_REPLAY_INDEX=1
export BALANCE_VALIDATION=0
export TRAIN_ITERS="${TRAIN_ITERS:-2}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export EVAL_ITERS="${EVAL_ITERS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"

exec bash "${SCRIPT_ROOT}/run_megatron_8gpu.sh"
