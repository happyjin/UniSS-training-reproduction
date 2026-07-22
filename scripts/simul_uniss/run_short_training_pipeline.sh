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

SHORT_RUN_DIR="${RUN_DIR}/short_training"
SHORT_LOG="${LOG_DIR}/short_training_pipeline.log"
GPU_SMOKE_MARKER="${RUN_DIR}/gpu_smoke/GPU_SMOKE_COMPLETE"

steps=(
  "reconstruct 16 source/target records"
  "train token streaming student for 100 steps"
  "train audio streaming student for 50 steps"
  "train action Qwen for 20 steps on 15-shard packed data"
  "train interleaved Qwen for 50 steps on 15-shard packed data"
  "train low-LR joint Qwen for 20 steps"
  "train BiCodec boundary refinement for 20 steps"
  "train GRPO policy for 50 SFT + 100 GRPO steps"
  "train NAR semantic generator for 100 steps"
  "run real BiCodec streaming replay"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${steps[@]}"
  exit 0
fi

for required in "${GPU_SMOKE_MARKER}" "${PACKED_TRAIN}" "${ACTION_PACKED_TRAIN}" "${VALID_PACKED_INTERLEAVED}" "${VALID_PACKED_ACTION}" "${POLICY_TOKENIZER_MODEL}"; do
  [[ -f "${required}" ]] || { echo "Missing short-training prerequisite: ${required}" >&2; exit 1; }
done

mkdir -p "${SHORT_RUN_DIR}" "${LOG_DIR}" "${SIMUL_CHECKPOINT_ROOT}"
{
  echo "[$(date -u +%FT%TZ)] starting Simul-UniSS 15-shard short training"
  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.reconstruct_unist_audio \
    --input "${UNIST_ROOT}/train-00000.parquet" \
    --output-dir "${STAGE0_AUDIO_DIR}" \
    --bicodec-model-dir "${BICODEC_MODEL_DIR}" \
    --device cuda:0 --limit-records 16 --side both

  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.train_streaming_student \
    --schedules "${SCHEDULES_JSONL}" \
    --policy-tokenizer "${POLICY_TOKENIZER_MODEL}" \
    --output-dir "${SIMUL_CHECKPOINT_ROOT}/stage1_bootstrap_student_short" \
    --tensorboard-dir "${TENSORBOARD_DIR}/short_stage1_token" \
    --qwen-checkpoint-root "${QWEN_CHECKPOINT_ROOT}" \
    --device cuda:0 --batch-size 4 --max-steps 100 \
    --hidden-size 128 --num-layers 2 --num-heads 8 --max-source-tokens 512 \
    --validation-records 128 --eval-interval 20 --eval-batches 4 --log-interval 5 --save-interval 20

  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.train_audio_student \
    --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl" \
    --policy-tokenizer "${POLICY_TOKENIZER_MODEL}" \
    --output-dir "${SIMUL_CHECKPOINT_ROOT}/stage1_audio_student_short" \
    --tensorboard-dir "${TENSORBOARD_DIR}/short_stage1_audio" \
    --device cuda:0 --batch-size 2 --max-steps 50 \
    --hidden-size 128 --num-layers 2 --num-heads 8 --max-audio-seconds 6 \
    --log-interval 5 --save-interval 10

  SIMUL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STAGE3_TRAIN_ITERS=20 \
    SIMUL_QWEN_SAVE_INTERVAL=10 SIMUL_QWEN_EVAL_INTERVAL=10 SIMUL_QWEN_EVAL_ITERS=2 \
    scripts/simul_uniss/train_qwen_stage.sh --stage action
  SIMUL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STAGE4_TRAIN_ITERS=50 \
    SIMUL_QWEN_SAVE_INTERVAL=10 SIMUL_QWEN_EVAL_INTERVAL=10 SIMUL_QWEN_EVAL_ITERS=2 \
    scripts/simul_uniss/train_qwen_stage.sh --stage interleaved
  SIMUL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STAGE6_TRAIN_ITERS=20 \
    SIMUL_QWEN_SAVE_INTERVAL=10 SIMUL_QWEN_EVAL_INTERVAL=10 SIMUL_QWEN_EVAL_ITERS=2 \
    scripts/simul_uniss/train_qwen_stage.sh --stage joint

  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.train_bicodec_refinement \
    --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl" \
    --bicodec-checkpoint "${BICODEC_MODEL_DIR}/BiCodec" \
    --output-dir "${SIMUL_CHECKPOINT_ROOT}/stage5_bicodec_refinement_short" \
    --tensorboard-dir "${TENSORBOARD_DIR}/short_stage5_bicodec" \
    --device cuda:0 --batch-size 1 --max-steps 20 --chunk-tokens 80 \
    --log-interval 2 --save-interval 10

  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.policy_grpo \
    --schedules "${SCHEDULES_JSONL}" \
    --output-dir "${SIMUL_CHECKPOINT_ROOT}/stage7_policy_grpo_short" \
    --tensorboard-dir "${TENSORBOARD_DIR}/short_stage7_grpo" \
    --device cuda:0 --batch-size 256 --sft-steps 50 --grpo-steps 100 \
    --group-size 8 --log-interval 5

  CUDA_VISIBLE_DEVICES=0 python -m training.simul_uniss.nar_semantic \
    --schedules "${SCHEDULES_JSONL}" \
    --output-dir "${SIMUL_CHECKPOINT_ROOT}/stage8_nar_semantic_short" \
    --tensorboard-dir "${TENSORBOARD_DIR}/short_stage8_nar" \
    --device cuda:0 --batch-size 8 --max-steps 100 \
    --hidden-size 128 --num-layers 2 --num-heads 8 --max-semantic-tokens 128 \
    --log-interval 5

  BICODEC_DEVICE=cuda:0 STAGE5_OUTPUT_DIR="${SHORT_RUN_DIR}/stage5_replay" \
    STAGE5_TENSORBOARD_DIR="${TENSORBOARD_DIR}/short_stage5_replay" \
    scripts/simul_uniss/run_stage5_streaming_replay.sh --decoder bicodec --record-index 0
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)" > "${SHORT_RUN_DIR}/SHORT_TRAINING_COMPLETE"
  echo "[$(date -u +%FT%TZ)] Simul-UniSS 15-shard short training complete"
} 2>&1 | tee -a "${SHORT_LOG}"
