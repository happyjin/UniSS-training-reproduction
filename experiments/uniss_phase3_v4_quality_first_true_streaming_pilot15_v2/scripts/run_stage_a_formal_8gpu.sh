#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

: "${FORMAL_TEACHER_ROOT:?FORMAL_TEACHER_ROOT is required}"
RUN_ID=${RUN_ID:-stage_a_v2_formal8_$(date -u +%Y%m%dT%H%M%SZ)}
export RUN_ID
export RUN_TRAIN_PACKS="${FORMAL_TRAIN_PACKS}"
export RUN_VALID_PACKS="${FORMAL_VALID_PACKS}"
export RUN_TRAIN_TEACHER_CACHE="${FORMAL_TEACHER_ROOT}/train/teacher_cache.jsonl"
export RUN_VALID_TEACHER_CACHE="${FORMAL_TEACHER_ROOT}/valid/teacher_cache.jsonl"
export RUN_LOAD="${PHASE3_NATIVE_ROOT}"
export RUN_SAVE_DIR="${CHECKPOINT_ROOT}/stage_a_formal/${RUN_ID}"
export RUN_TENSORBOARD_DIR="${RUN_ROOT}/stage_a_formal/${RUN_ID}/tensorboard"
export RUN_LOG="${LOG_ROOT}/stage_a_formal/${RUN_ID}/train.log"
export RUN_SEQ_LENGTH=18000
export RUN_MBS=1
export RUN_GBS=128
export RUN_COVERAGE_EPOCHS=3
export RUN_TRAIN_ITERS=381
export RUN_MAX_ACOUSTICS=2
export RUN_NUM_WORKERS=4
export RUN_MASTER_PORT=${RUN_MASTER_PORT:-29723}
export RUN_SAVE_INTERVAL=50
export RUN_EVAL_INTERVAL=50
export RUN_EVAL_ITERS=1
export RUN_LOG_INTERVAL=1
export RUN_WARMUP_ITERS=19
export RUN_STRICTNESS=log_all
export RUN_SMOKE=0
export RUN_AUDIT_GRADIENTS=0
export RUN_FINETUNE=1
export RUN_LOAD_OPTIM=0
export RUN_LOAD_RNG=0

exec bash "${SCRIPT_DIR}/run_stage_a_megatron.sh" "$@"
