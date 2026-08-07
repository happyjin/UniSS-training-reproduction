#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="${EVAL_ROOT}/../scripts"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/stage_b_env.sh"

for name in CHECKPOINT_DIR MODEL_LABEL CHUNK OUTPUT_LOG; do
  [[ -n "${!name:-}" ]] || { echo "${name} must be set" >&2; exit 1; }
done
case "${CHUNK}" in
  320|640|960|1280|offline) ;;
  *) echo "CHUNK must be one of 320,640,960,1280,offline" >&2; exit 1 ;;
esac

require_dir "${CHECKPOINT_DIR}"
require_file "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
require_file "${PILOT_ROOT}/joint_train.jsonl"
require_file "${PILOT_ROOT}/joint_valid.jsonl"
require_file "${PILOT_REPLAY_OFFSETS}"
require_file "${PILOT_REPLAY_OFFSETS}.json"
require_dir "${PILOT_ROOT}/tokenizer_maps"
require_dir "${PILOT_ROOT}/direction_indices"
require_dir "${WHISPER_MODEL}"
require_dir "${PHASE3_MODEL}"
refuse_existing "${OUTPUT_LOG}"
mkdir -p "$(dirname "${OUTPUT_LOG}")"

export CUDA_VISIBLE_DEVICES
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-29765}" \
  "${REPO_ROOT}/training/phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py" \
  --joint-train-manifest "${PILOT_ROOT}/joint_train.jsonl" \
  --joint-valid-manifest "${PILOT_ROOT}/joint_valid.jsonl" \
  --joint-tokenizer-map-dir "${PILOT_ROOT}/tokenizer_maps" \
  --joint-direction-index-dir "${PILOT_ROOT}/direction_indices" \
  --joint-phase3-replay-packed "${PHASE3_REPLAY_PACKED}" \
  --joint-phase3-replay-offsets "${PILOT_REPLAY_OFFSETS}" \
  --joint-whisper-model "${WHISPER_MODEL}" \
  --joint-phase3-model "${PHASE3_MODEL}" \
  --joint-chunks "${CHUNK}" \
  --joint-right-context-ms 80 \
  --joint-replay-probability 0.20 \
  --joint-bicodec-ctc-weight "${BICODEC_WEIGHT}" \
  --joint-ar-s2tt-weight "${AR_WEIGHT}" \
  --joint-asr-ctc-weight "${ASR_WEIGHT}" \
  --joint-nar-s2tt-ctc-weight "${NAR_WEIGHT}" \
  --joint-phase3-replay-weight "${REPLAY_WEIGHT}" \
  --joint-bridge-commitment-weight "${BRIDGE_COMMITMENT_WEIGHT}" \
  --joint-whisper-quantize-weight "${WHISPER_QUANTIZE_WEIGHT}" \
  --joint-teacher-glm-ce-weight "${TEACHER_CE_WEIGHT}" \
  --joint-teacher-glm-commitment-weight "${TEACHER_COMMITMENT_WEIGHT}" \
  --joint-unit-upsample-ratio 48 \
  --joint-bridge-surrogate topk_soft \
  --joint-bridge-topk 8 \
  --joint-bridge-temperature 0.1 \
  --joint-bridge-gradient-scale "${BRIDGE_GRADIENT_SCALE}" \
  --joint-teacher-temperature "${TEACHER_TEMPERATURE:-0.1}" \
  --joint-bridge-guard-baseline-microbatches "${BASELINE_MICROBATCHES}" \
  --joint-bridge-guard-relative-consecutive-violations "${GUARD_CONSECUTIVE_VIOLATIONS}" \
  --joint-max-bridge-commitment "${MAX_COMMITMENT}" \
  --joint-max-bridge-commitment-ratio "${MAX_COMMITMENT_RATIO}" \
  --joint-freeze-whisper-codebook \
  --joint-freeze-whisper-post-vq \
  --joint-trainable-whisper-pre-vq-layers "${TRAINABLE_WHISPER_LAYERS}" \
  --joint-lr-new-mult "${LR_NEW_MULT}" \
  --joint-lr-bridge-mult 0 \
  --joint-lr-whisper-top-mult "${LR_WHISPER_TOP_MULT}" \
  --joint-lr-whisper-bottom-mult 0 \
  --joint-lr-qwen-mult "${LR_QWEN_MULT}" \
  --joint-lr-qwen-io-mult "${LR_QWEN_IO_MULT}" \
  --joint-allow-partial-replay-index \
  --joint-balance-validation \
  --load "${CHECKPOINT_DIR}" \
  --skip-train \
  --no-load-optim \
  --no-load-rng \
  --tokenizer-type NullTokenizer \
  --vocab-size 180407 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 896 \
  --ffn-hidden-size 4864 \
  --num-attention-heads 14 \
  --group-query-attention \
  --num-query-groups 2 \
  --normalization RMSNorm \
  --swiglu \
  --disable-bias-linear \
  --add-qkv-bias \
  --position-embedding-type rope \
  --rotary-base 1000000 \
  --bf16 \
  --seq-length 18000 \
  --max-position-embeddings 32768 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters 500 \
  --lr "${BASE_LR}" \
  --min-lr "${MIN_LR}" \
  --lr-warmup-iters 1 \
  --lr-decay-iters 500 \
  --lr-decay-style inverse-square-root \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --adam-eps 1e-8 \
  --clip-grad 0.5 \
  --dataloader-type cyclic \
  --no-data-sharding \
  --num-workers "${NUM_WORKERS}" \
  --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --eval-iters "${EVAL_ITERS:-8}" \
  --log-interval 1 \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_LOG}"

grep -q "validation loss at iteration .* on validation set" "${OUTPUT_LOG}" || {
  echo "${MODEL_LABEL}/${CHUNK}: final validation record is missing" >&2
  exit 1
}
