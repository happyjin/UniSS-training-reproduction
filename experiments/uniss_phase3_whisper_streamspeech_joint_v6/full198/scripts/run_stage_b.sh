#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
REQUESTED_BASELINE_MICROBATCHES="${BASELINE_MICROBATCHES:-}"
REQUESTED_GUARD_CONSECUTIVE_VIOLATIONS="${GUARD_CONSECUTIVE_VIOLATIONS:-}"
# shellcheck source=/dev/null
source "${V6_EXPERIMENT_ROOT}/scripts/stage_b_env.sh"
[[ -n "${REQUESTED_BASELINE_MICROBATCHES}" ]] && export BASELINE_MICROBATCHES="${REQUESTED_BASELINE_MICROBATCHES}"
[[ -n "${REQUESTED_GUARD_CONSECUTIVE_VIOLATIONS}" ]] && export GUARD_CONSECUTIVE_VIOLATIONS="${REQUESTED_GUARD_CONSECUTIVE_VIOLATIONS}"

validate_full198_data
export_full198_inputs
export MICRO_BATCH_SIZE="${STAGE_B_MICRO_BATCH_SIZE}"
export RUN_NAME="${RUN_NAME:-${STAGE_B_RUN_NAME}}"
export LOAD_DIR="${LOAD_DIR:-${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/${STAGE_A_RUN_NAME}}"
export TRAIN_ITERS="${TRAIN_ITERS:-9075}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-400}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-8}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"

exec bash "${V6_EXPERIMENT_ROOT}/scripts/run_stage_8gpu.sh"
