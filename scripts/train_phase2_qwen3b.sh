#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/neuhao}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/runs"

TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/data/megatron/phase2_qwen3b_qp_replay/packed_train.jsonl}"
VALID_DATA="${VALID_DATA:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_qwen3b_phase1_unist198_recovery_from7000_v1}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3b_phase2_qp_replay}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29731}"
LOAD_OPTIM="${LOAD_OPTIM:-0}"
LOAD_RNG="${LOAD_RNG:-0}"
FINETUNE="${FINETUNE:-1}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-}"
DATALOADER_TYPE="${DATALOADER_TYPE:-cyclic}"
CLIP_GRAD="${CLIP_GRAD:-0.5}"

if [[ "${DRY_RUN}" == "0" && ! -f "${TRAIN_DATA}" ]]; then
  echo "Missing TRAIN_DATA: ${TRAIN_DATA}" >&2
  exit 1
fi

cmd=(torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/training/pretrain_uniss_megatron.py"
  --sft
  --uniss-packed-train "${TRAIN_DATA}"
  --uniss-strict-paper-config
  --tokenizer-type NullTokenizer
  --vocab-size 180407
  --tensor-model-parallel-size "${TP:-2}"
  --pipeline-model-parallel-size "${PP:-1}"
  --num-layers 36
  --hidden-size 2048
  --ffn-hidden-size 11008
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 2
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --add-qkv-bias
  --position-embedding-type rope
  --rotary-base 1000000
  --seq-length 18000
  --max-position-embeddings 32768
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --global-batch-size 128
  --train-iters "${TRAIN_ITERS:-1}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-warmup-iters "${LR_WARMUP_ITERS:-400}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --dataloader-type "${DATALOADER_TYPE}"
  --clip-grad "${CLIP_GRAD}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --use-flash-attn
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --save "${SAVE_DIR}"
  --load "${LOAD_CHECKPOINT}"
  --save-interval "${SAVE_INTERVAL:-100}"
  --log-interval "${LOG_INTERVAL:-10}"
)

if [[ -n "${LR_DECAY_ITERS}" ]]; then
  cmd+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${LOAD_OPTIM}" != "1" ]]; then
  cmd+=(--no-load-optim)
fi
if [[ "${LOAD_RNG}" != "1" ]]; then
  cmd+=(--no-load-rng)
fi
if [[ "${FINETUNE}" == "1" ]]; then
  cmd+=(--finetune)
fi
if [[ -n "${VALID_DATA}" ]]; then
  cmd+=(--uniss-packed-valid "${VALID_DATA}" --eval-iters "${EVAL_ITERS:-10}" --eval-interval "${EVAL_INTERVAL:-100}")
else
  cmd+=(--eval-iters 0)
fi
if [[ -n "${TENSORBOARD_DIR:-}" ]]; then
  mkdir -p "${TENSORBOARD_DIR}"
  cmd+=(
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL:-10}"
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-memory-interval "${TENSORBOARD_MEMORY_INTERVAL:-10}"
    --log-world-size-to-tensorboard
    --log-throughput
  )
fi

cmd+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
