#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

: "${RUN_NAME:?RUN_NAME must be set}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST must be set}"
: "${VALID_MANIFEST:?VALID_MANIFEST must be set}"
: "${TOKENIZER_MAP_DIR:?TOKENIZER_MAP_DIR must be set}"
: "${DIRECTION_INDEX_DIR:?DIRECTION_INDEX_DIR must be set}"
: "${REPLAY_OFFSETS:?REPLAY_OFFSETS must be set}"
: "${TRAIN_ITERS:?TRAIN_ITERS must be set}"

SAVE_DIR="${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v1/${RUN_NAME}"
TB_DIR="${REPO_ROOT}/runs/uniss_phase3_whisper_streamspeech_joint_v1/${RUN_NAME}"
LOG="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/${RUN_NAME}.log"
refuse_existing "${SAVE_DIR}" "${TB_DIR}" "${LOG}"
require_file "${TRAIN_MANIFEST}"
require_file "${VALID_MANIFEST}"
require_file "${REPLAY_OFFSETS}"
require_file "${REPLAY_OFFSETS}.json"
require_dir "${TOKENIZER_MAP_DIR}"
require_dir "${DIRECTION_INDEX_DIR}"
require_dir "${WHISPER_MODEL}"
require_dir "${PHASE3_MODEL}"

EXTRA_REPLAY_ARGS=()
if [[ "${ALLOW_PARTIAL_REPLAY_INDEX:-0}" == "1" ]]; then
  EXTRA_REPLAY_ARGS+=(--joint-allow-partial-replay-index)
fi
EXTRA_VALID_ARGS=()
if [[ "${BALANCE_VALIDATION:-0}" == "1" ]]; then
  EXTRA_VALID_ARGS+=(--joint-balance-validation)
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
  --joint-bicodec-ctc-weight 1 \
  --joint-ar-s2tt-weight 8 \
  --joint-asr-ctc-weight 4 \
  --joint-nar-s2tt-ctc-weight 4 \
  --joint-phase3-replay-weight 0.5 \
  --joint-unit-upsample-ratio 48 \
  "${EXTRA_REPLAY_ARGS[@]}" \
  "${EXTRA_VALID_ARGS[@]}" \
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
  --seq-length 18000 \
  --max-position-embeddings 32768 \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-warmup-iters "${LR_WARMUP_ITERS:-4000}" \
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
