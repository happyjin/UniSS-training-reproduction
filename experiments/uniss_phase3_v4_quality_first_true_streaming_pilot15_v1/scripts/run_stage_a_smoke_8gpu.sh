#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

RUN_ID=${RUN_ID:-stage_a_smoke8_$(date -u +%Y%m%dT%H%M%SZ)}
SMOKE_ROOT="${STAGE_A_DATA_ROOT}/pack_smoke_20260816T070000Z"
export RUN_ID
export RUN_TRAIN_PACKS="${SMOKE_ROOT}/train_packs.jsonl"
export RUN_VALID_PACKS="${SMOKE_ROOT}/valid_packs.jsonl"
export RUN_SAVE_DIR="${CHECKPOINT_ROOT}/stage_a_smoke/${RUN_ID}"
export RUN_TENSORBOARD_DIR="${RUN_ROOT}/stage_a_smoke/${RUN_ID}/tensorboard"
export RUN_LOG="${LOG_ROOT}/stage_a_smoke/${RUN_ID}/train.log"
export RUN_SEQ_LENGTH=4096
export RUN_MBS=1
export RUN_GBS=16
export RUN_COVERAGE_EPOCHS=32
export RUN_TRAIN_ITERS=32
export RUN_MAX_ACOUSTICS=1
export RUN_NUM_WORKERS=2
export RUN_SAVE_INTERVAL=5
export RUN_EVAL_INTERVAL=10
export RUN_EVAL_ITERS=1
export RUN_LOG_INTERVAL=1
export RUN_WARMUP_ITERS=1
export RUN_SMOKE=1
export RUN_AUDIT_GRADIENTS=1

exec bash "${SCRIPT_DIR}/run_stage_a_megatron.sh" "$@"
