#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"

: "${SMOKE_TEACHER_ROOT:?SMOKE_TEACHER_ROOT is required}"
: "${RESUME_LOAD:?RESUME_LOAD must point to a Stage A v2 checkpoint root}"
RUN_ID=${RUN_ID:-stage_a_v2_strict_resume8_$(date -u +%Y%m%dT%H%M%SZ)}
export RUN_ID
export RUN_TRAIN_PACKS="${SMOKE_TRAIN_PACKS}"
export RUN_VALID_PACKS="${SMOKE_VALID_PACKS}"
export RUN_TRAIN_TEACHER_CACHE="${SMOKE_TEACHER_ROOT}/train/teacher_cache.jsonl"
export RUN_VALID_TEACHER_CACHE="${SMOKE_TEACHER_ROOT}/valid/teacher_cache.jsonl"
export RUN_LOAD="${RESUME_LOAD}"
export RUN_SAVE_DIR="${CHECKPOINT_ROOT}/stage_a_strict_resume/${RUN_ID}"
export RUN_TENSORBOARD_DIR="${RUN_ROOT}/stage_a_strict_resume/${RUN_ID}/tensorboard"
export RUN_LOG="${LOG_ROOT}/stage_a_strict_resume/${RUN_ID}/train.log"
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
export RUN_STRICTNESS=raise_all
export RUN_SMOKE=1
export RUN_AUDIT_GRADIENTS=1
export RUN_FINETUNE=0
export RUN_LOAD_OPTIM=1
export RUN_LOAD_RNG=1
export RUN_SKIP_SAVE=1
export RUN_MASTER_PORT=${RUN_MASTER_PORT:-29724}

exec bash "${SCRIPT_DIR}/run_stage_a_megatron.sh" "$@"
