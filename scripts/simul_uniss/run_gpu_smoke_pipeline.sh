#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

SMOKE_ROOT="${SIMUL_CHECKPOINT_ROOT}/gpu_smoke"
SMOKE_RUN="${RUN_DIR}/gpu_smoke"
SMOKE_SCHEDULES="${REPO_ROOT}/data/processed/simul_uniss_v1/smoke/schedules.jsonl"
SMOKE_PACKED="${REPO_ROOT}/data/megatron/simul_uniss_v1/smoke/packed_train.jsonl"
SMOKE_ACTION_PACKED="${REPO_ROOT}/data/megatron/simul_uniss_v1/smoke/packed_action_train.jsonl"
PHASE1_COMPLETE="${QWEN_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

steps=(
  "scripts/simul_uniss/run_stage0_prefix_baseline.sh --records 1"
  "scripts/simul_uniss/train_stage1_audio_student.sh --smoke"
  "python -m training.simul_uniss.train_streaming_student --schedules ${SMOKE_SCHEDULES} --policy-tokenizer ${POLICY_TOKENIZER_MODEL} --output-dir ${SMOKE_ROOT}/stage1_token --tensorboard-dir ${TENSORBOARD_DIR}/gpu_smoke_stage1_token --qwen-checkpoint-root ${QWEN_CHECKPOINT_ROOT} --device cuda:0 --batch-size 1 --max-steps 2 --hidden-size 64 --num-layers 2 --num-heads 4 --max-source-tokens 128 --validation-records 4 --eval-interval 1 --eval-batches 1 --log-interval 1 --save-interval 1"
  "scripts/simul_uniss/train_qwen_stage.sh --stage action --smoke"
  "scripts/simul_uniss/train_qwen_stage.sh --stage interleaved --smoke"
  "scripts/simul_uniss/train_qwen_stage.sh --stage joint --smoke"
  "scripts/simul_uniss/train_stage5_bicodec_refinement.sh --smoke"
  "scripts/simul_uniss/run_stage5_streaming_replay.sh --decoder bicodec --record-index 0"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${steps[@]}"
  exit 0
fi

iteration="$(tr -d '[:space:]' < "${PHASE1_COMPLETE}")"
if [[ ! "${iteration}" =~ ^[0-9]+$ ]] || (( iteration < 15465 )); then
  echo "Phase1 recovery is not complete: iteration=${iteration}" >&2
  exit 1
fi
for required in "${POLICY_TOKENIZER_MODEL}" "${SMOKE_SCHEDULES}" "${SMOKE_PACKED}" "${SMOKE_ACTION_PACKED}" "${VALID_PACKED_INTERLEAVED}" "${VALID_PACKED_ACTION}"; do
  [[ -f "${required}" ]] || { echo "Missing smoke prerequisite: ${required}" >&2; exit 1; }
done

mkdir -p "${SMOKE_ROOT}" "${SMOKE_RUN}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=0
export SIMUL_CUDA_VISIBLE_DEVICES=0
export STAGE0_DEVICE=cuda:0
export STAGE1_AUDIO_DEVICE=cuda:0
export STAGE1_AUDIO_OUTPUT_DIR="${SMOKE_ROOT}/stage1_audio"
export STAGE1_AUDIO_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage1_audio"
export PACKED_TRAIN="${SMOKE_PACKED}"
export ACTION_PACKED_TRAIN="${SMOKE_ACTION_PACKED}"
export SCHEDULES_JSONL="${SMOKE_SCHEDULES}"
export STAGE3_SAVE_ROOT="${SMOKE_ROOT}/stage3_action"
export STAGE3_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage3_action"
export STAGE4_LOAD_ROOT="${STAGE3_SAVE_ROOT}"
export STAGE4_SAVE_ROOT="${SMOKE_ROOT}/stage4_interleaved"
export STAGE4_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage4_interleaved"
export STAGE6_LOAD_ROOT="${STAGE4_SAVE_ROOT}"
export STAGE6_SAVE_ROOT="${SMOKE_ROOT}/stage6_joint"
export STAGE6_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage6_joint"
export STAGE5_OUTPUT_DIR="${SMOKE_RUN}/stage5_bicodec"
export STAGE5_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage5_bicodec"
export STAGE5_REFINEMENT_OUTPUT_DIR="${SMOKE_ROOT}/stage5_bicodec_refinement"
export STAGE5_REFINEMENT_TENSORBOARD_DIR="${TENSORBOARD_DIR}/gpu_smoke_stage5_bicodec_refinement"
export BICODEC_DEVICE=cuda:0

{
  echo "[$(date -u +%FT%TZ)] starting Simul-UniSS GPU smoke from Phase1 iteration ${iteration}"
  scripts/simul_uniss/run_stage0_prefix_baseline.sh --records 1
  scripts/simul_uniss/train_stage1_audio_student.sh --smoke
  python -m training.simul_uniss.train_streaming_student \
    --schedules "${SMOKE_SCHEDULES}" \
    --policy-tokenizer "${POLICY_TOKENIZER_MODEL}" \
    --output-dir "${SMOKE_ROOT}/stage1_token" \
    --tensorboard-dir "${TENSORBOARD_DIR}/gpu_smoke_stage1_token" \
    --qwen-checkpoint-root "${QWEN_CHECKPOINT_ROOT}" \
    --device cuda:0 --batch-size 1 --max-steps 2 \
    --hidden-size 64 --num-layers 2 --num-heads 4 --max-source-tokens 128 \
    --validation-records 4 --eval-interval 1 --eval-batches 1 --log-interval 1 --save-interval 1
  scripts/simul_uniss/train_qwen_stage.sh --stage action --smoke
  scripts/simul_uniss/train_qwen_stage.sh --stage interleaved --smoke
  scripts/simul_uniss/train_qwen_stage.sh --stage joint --smoke
  scripts/simul_uniss/train_stage5_bicodec_refinement.sh --smoke
  scripts/simul_uniss/run_stage5_streaming_replay.sh --decoder bicodec --record-index 0
  printf 'completed_at=%s\nphase1_anchor=%s\n' "$(date -u +%FT%TZ)" "${iteration}" > "${SMOKE_RUN}/GPU_SMOKE_COMPLETE"
  echo "[$(date -u +%FT%TZ)] Simul-UniSS GPU smoke complete"
} 2>&1 | tee -a "${LOG_DIR}/gpu_smoke_pipeline.log"
