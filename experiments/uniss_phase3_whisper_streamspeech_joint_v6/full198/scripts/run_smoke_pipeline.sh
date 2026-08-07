#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

SMOKE_TAG="${SMOKE_TAG:-v2}"
SMOKE_A="phase3_joint_v6_stage_a_heads_only_full198_mbs2_smoke_${SMOKE_TAG}"
SMOKE_B="phase3_joint_v6_stage_b_guarded_joint_full198_mbs1_smoke_${SMOKE_TAG}"

RUN_NAME="${SMOKE_A}" \
TRAIN_ITERS=4 LR_WARMUP_ITERS=2 SAVE_INTERVAL=4 EVAL_INTERVAL=4 EVAL_ITERS=2 LOG_INTERVAL=1 \
bash "${SCRIPT_ROOT}/run_stage_a.sh"

RUN_NAME="${SMOKE_B}" \
LOAD_DIR="${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/${SMOKE_A}" \
STAGE_B_MICRO_BATCH_SIZE=1 \
TRAIN_ITERS=16 LR_WARMUP_ITERS=4 SAVE_INTERVAL=16 EVAL_INTERVAL=8 EVAL_ITERS=2 LOG_INTERVAL=1 \
BASELINE_MICROBATCHES=8 GUARD_CONSECUTIVE_VIOLATIONS=4 \
bash "${SCRIPT_ROOT}/run_stage_b.sh"

echo "full198 Stage A MBS=2 -> Stage B MBS=1 smoke completed"
