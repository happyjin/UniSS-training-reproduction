#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/stage_a_env.sh"
export RUN_NAME="${RUN_NAME:-phase3_joint_v6_stage_a_heads_only_15shard_v2}"
export TRAIN_MANIFEST="${PILOT_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${PILOT_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${PILOT_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${PILOT_ROOT}/direction_indices"
export REPLAY_OFFSETS="${PILOT_REPLAY_OFFSETS}"
export BALANCE_VALIDATION=1
export TRAIN_ITERS="${TRAIN_ITERS:-200}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-50}"
exec bash "${SCRIPT_ROOT}/run_stage_8gpu.sh"
