#!/usr/bin/env bash
set -euo pipefail

STAGE=""
DRY_RUN=0
SMOKE=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${STAGE}" != "action" && "${STAGE}" != "interleaved" && "${STAGE}" != "joint" ]]; then
  echo "--stage must be action, interleaved, or joint" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

if [[ "${STAGE}" == "action" ]]; then
  TRAIN_DATA="${ACTION_PACKED_TRAIN}"
  LOAD_ROOT="${STAGE3_LOAD_ROOT}"
  SAVE_ROOT="${STAGE3_SAVE_ROOT}"
  MASTER_PORT="${STAGE3_MASTER_PORT}"
  TRAIN_ITERS="${STAGE3_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE3_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_ACTION}"
elif [[ "${STAGE}" == "interleaved" ]]; then
  TRAIN_DATA="${PACKED_TRAIN}"
  LOAD_ROOT="${STAGE4_LOAD_ROOT}"
  SAVE_ROOT="${STAGE4_SAVE_ROOT}"
  MASTER_PORT="${STAGE4_MASTER_PORT}"
  TRAIN_ITERS="${STAGE4_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE4_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_INTERLEAVED}"
else
  TRAIN_DATA="${PACKED_TRAIN}"
  LOAD_ROOT="${STAGE6_LOAD_ROOT}"
  SAVE_ROOT="${STAGE6_SAVE_ROOT}"
  MASTER_PORT="${STAGE6_MASTER_PORT}"
  TRAIN_ITERS="${STAGE6_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE6_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_INTERLEAVED}"
fi

NPROC="${SIMUL_NPROC_PER_NODE}"
MICRO_BATCH="${SIMUL_MICRO_BATCH_SIZE}"
GLOBAL_BATCH="${SIMUL_GLOBAL_BATCH_SIZE}"
WARMUP_ITERS="${SIMUL_QWEN_WARMUP_ITERS}"
SAVE_INTERVAL="${SIMUL_QWEN_SAVE_INTERVAL}"
EVAL_INTERVAL="${SIMUL_QWEN_EVAL_INTERVAL}"
EVAL_ITERS="${SIMUL_QWEN_EVAL_ITERS}"
QWEN_LR="${SIMUL_QWEN_LR}"
QWEN_MIN_LR="${SIMUL_QWEN_MIN_LR}"
if [[ "${STAGE}" == "joint" ]]; then
  QWEN_LR="${STAGE6_QWEN_LR}"
  QWEN_MIN_LR="${STAGE6_QWEN_MIN_LR}"
fi
if [[ "${SMOKE}" == "1" ]]; then
  NPROC=1
  MICRO_BATCH=1
  GLOBAL_BATCH=1
  TRAIN_ITERS=2
  WARMUP_ITERS=0
  SAVE_INTERVAL=1
  EVAL_INTERVAL=1
  EVAL_ITERS=1
fi

if [[ "${DRY_RUN}" == "0" && ! -f "${TRAIN_DATA}" ]]; then
  echo "Missing training data: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "0" && ! -f "${LOAD_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing load checkpoint pointer: ${LOAD_ROOT}/latest_checkpointed_iteration.txt" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "0" && ! -f "${VALID_DATA}" ]]; then
  echo "Missing validation data: ${VALID_DATA}" >&2
  exit 1
fi

cmd=(torchrun
  --nproc_per_node "${NPROC}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/training/pretrain_simul_uniss_megatron.py"
  --sft
  --simul-packed-train "${TRAIN_DATA}"
  --simul-schema-version simul_uniss_packed_v1
  --tokenizer-type NullTokenizer
  --vocab-size 180407
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers 24
  --hidden-size 896
  --ffn-hidden-size 4864
  --num-attention-heads 14
  --group-query-attention
  --num-query-groups 2
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --add-qkv-bias
  --position-embedding-type rope
  --rotary-base 1000000
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings 32768
  --micro-batch-size "${MICRO_BATCH}"
  --global-batch-size "${GLOBAL_BATCH}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${QWEN_LR}"
  --min-lr "${QWEN_MIN_LR}"
  --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_ITERS}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --save "${SAVE_ROOT}"
  --load "${LOAD_ROOT}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval 1
  --simul-packed-valid "${VALID_DATA}"
  --eval-iters "${EVAL_ITERS}"
  --eval-interval "${EVAL_INTERVAL}"
  --no-load-optim
  --no-load-rng
  --finetune
  --tensorboard-dir "${STAGE_TENSORBOARD_DIR}"
  --tensorboard-log-interval 1
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval 1
  --log-throughput
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${SAVE_ROOT}" "${STAGE_TENSORBOARD_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage_${STAGE}_qwen.log"
