#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

for name in RUN_NAME TRAIN_MANIFEST VALID_MANIFEST TOKENIZER_MAP_DIR DIRECTION_INDEX_DIR REPLAY_OFFSETS TRAIN_ITERS BALANCE_VALIDATION; do
  [[ -n "${!name:-}" ]] || { echo "${name} must be set" >&2; exit 1; }
done

SAVE_DIR="${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/${RUN_NAME}"
TB_DIR="${REPO_ROOT}/runs/uniss_phase3_whisper_streamspeech_joint_v6/${RUN_NAME}"
LOG="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v6/${RUN_NAME}.log"
refuse_existing "${SAVE_DIR}" "${TB_DIR}" "${LOG}"
require_file "${TRAIN_MANIFEST}"
require_file "${VALID_MANIFEST}"
require_file "${REPLAY_OFFSETS}"
require_file "${REPLAY_OFFSETS}.json"
require_dir "${TOKENIZER_MAP_DIR}"
require_dir "${DIRECTION_INDEX_DIR}"
require_dir "${WHISPER_MODEL}"
require_dir "${PHASE3_MODEL}"

EXTRA_ARGS=(--joint-allow-partial-replay-index)
[[ "${BALANCE_VALIDATION}" == "1" ]] && EXTRA_ARGS+=(--joint-balance-validation)
[[ "${FREEZE_WHISPER:-0}" == "1" ]] && EXTRA_ARGS+=(--joint-freeze-whisper)
[[ "${FREEZE_QWEN:-0}" == "1" ]] && EXTRA_ARGS+=(--joint-freeze-qwen)
if [[ -n "${LOAD_DIR:-}" ]]; then
  require_dir "${LOAD_DIR}"
  EXTRA_ARGS+=(--load "${LOAD_DIR}" --finetune --no-load-optim --no-load-rng)
fi

export CUDA_VISIBLE_DEVICES
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/training/phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py" \
  --joint-train-manifest "${TRAIN_MANIFEST}" \
  --joint-valid-manifest "${VALID_MANIFEST}" \
  --joint-tokenizer-map-dir "${TOKENIZER_MAP_DIR}" \
  --joint-direction-index-dir "${DIRECTION_INDEX_DIR}" \
  --joint-phase3-replay-packed "${PHASE3_REPLAY_PACKED}" \
  --joint-phase3-replay-offsets "${REPLAY_OFFSETS}" \
  --joint-whisper-model "${WHISPER_MODEL}" \
  --joint-phase3-model "${PHASE3_MODEL}" \
  --joint-chunks "320,640,960,1280,offline" \
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
  --joint-max-bridge-commitment "${MAX_COMMITMENT}" \
  --joint-max-bridge-commitment-ratio "${MAX_COMMITMENT_RATIO}" \
  --joint-bridge-guard-baseline-microbatches "${BASELINE_MICROBATCHES}" \
  --joint-freeze-whisper-codebook \
  --joint-freeze-whisper-post-vq \
  --joint-trainable-whisper-pre-vq-layers "${TRAINABLE_WHISPER_LAYERS}" \
  --joint-lr-new-mult "${LR_NEW_MULT}" \
  --joint-lr-bridge-mult 0 \
  --joint-lr-whisper-top-mult "${LR_WHISPER_TOP_MULT}" \
  --joint-lr-whisper-bottom-mult 0 \
  --joint-lr-qwen-mult "${LR_QWEN_MULT}" \
  --joint-lr-qwen-io-mult "${LR_QWEN_IO_MULT}" \
  "${EXTRA_ARGS[@]}" \
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
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr "${BASE_LR}" \
  --min-lr "${MIN_LR}" \
  --lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --lr-decay-iters "${TRAIN_ITERS}" \
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
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-100}" \
  --eval-interval "${EVAL_INTERVAL:-100}" \
  --eval-iters "${EVAL_ITERS:-8}" \
  --log-interval "${LOG_INTERVAL:-10}" \
  --tensorboard-dir "${TB_DIR}" \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --seed "${SEED}" \
  2>&1 | tee "${LOG}"
