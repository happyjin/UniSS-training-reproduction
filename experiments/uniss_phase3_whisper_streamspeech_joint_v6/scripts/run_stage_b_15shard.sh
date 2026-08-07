#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/stage_b_env.sh"
export RUN_NAME="${RUN_NAME:-phase3_joint_v6_stage_b_guarded_joint_15shard_v1}"
export TRAIN_MANIFEST="${PILOT_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${PILOT_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${PILOT_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${PILOT_ROOT}/direction_indices"
export REPLAY_OFFSETS="${PILOT_REPLAY_OFFSETS}"
export BALANCE_VALIDATION=1
export LOAD_DIR="${LOAD_DIR:-${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_a_heads_only_15shard_v1}"
export TRAIN_ITERS="${TRAIN_ITERS:-500}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-100}"
exec bash "${SCRIPT_ROOT}/run_stage_8gpu.sh"
