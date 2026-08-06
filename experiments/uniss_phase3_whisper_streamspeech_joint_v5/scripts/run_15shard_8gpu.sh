#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

require_file "${PILOT_ROOT}/manifest_summary.json"
export RUN_NAME="${RUN_NAME:-phase3_joint_v5_15shard_guarded_v1}"
export TRAIN_MANIFEST="${PILOT_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${PILOT_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${PILOT_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${PILOT_ROOT}/direction_indices"
export REPLAY_OFFSETS="${PILOT_REPLAY_OFFSETS}"
export ALLOW_PARTIAL_REPLAY_INDEX=1
export BALANCE_VALIDATION=1
export TRAIN_ITERS="${TRAIN_ITERS:-1500}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-300}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-8}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"

exec bash "${SCRIPT_ROOT}/run_megatron_8gpu.sh"
