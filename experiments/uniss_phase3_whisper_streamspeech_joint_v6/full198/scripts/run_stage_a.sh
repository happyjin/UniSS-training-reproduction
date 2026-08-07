#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
# shellcheck source=/dev/null
source "${V6_EXPERIMENT_ROOT}/scripts/stage_a_env.sh"

validate_full198_data
export_full198_inputs
export MICRO_BATCH_SIZE="${STAGE_A_MICRO_BATCH_SIZE}"
export RUN_NAME="${RUN_NAME:-${STAGE_A_RUN_NAME}}"
export TRAIN_ITERS="${TRAIN_ITERS:-500}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-8}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
unset LOAD_DIR

exec bash "${V6_EXPERIMENT_ROOT}/scripts/run_stage_8gpu.sh"
