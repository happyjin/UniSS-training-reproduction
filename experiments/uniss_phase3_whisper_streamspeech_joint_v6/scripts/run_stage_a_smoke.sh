#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/stage_a_env.sh"
export RUN_NAME="${RUN_NAME:-phase3_joint_v6_stage_a_heads_only_smoke_v3}"
export TRAIN_MANIFEST="${SMOKE_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${SMOKE_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${SMOKE_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${SMOKE_ROOT}/direction_indices"
export REPLAY_OFFSETS="${SMOKE_REPLAY_OFFSETS}"
export BALANCE_VALIDATION=0
export TRAIN_ITERS="${TRAIN_ITERS:-2}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1}"
export SAVE_INTERVAL=1 EVAL_INTERVAL=1 EVAL_ITERS=1 LOG_INTERVAL=1
exec bash "${SCRIPT_ROOT}/run_stage_8gpu.sh"
