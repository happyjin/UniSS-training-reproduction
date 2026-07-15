#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/runs"

TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/data/megatron/phase3/packed_train.jsonl}"
VALID_DATA="${VALID_DATA:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_phase2}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_phase3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MASTER_PORT="${MASTER_PORT:-29503}"

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
  --tensor-model-parallel-size "${TP:-1}"
  --pipeline-model-parallel-size "${PP:-1}"
  --num-layers 28
  --hidden-size 1536
  --ffn-hidden-size 8960
  --num-attention-heads 12
  --group-query-attention
  --num-query-groups 2
  --normalization RMSNorm
  --swiglu
  --position-embedding-type rope
  --rotary-base 1000000
  --seq-length 18000
  --max-position-embeddings 32768
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --global-batch-size 128
  --train-iters "${TRAIN_ITERS:-4341}"
  --lr 5e-5
  --min-lr 5e-6
  --lr-warmup-iters "${LR_WARMUP_ITERS:-0}"
  --lr-decay-style cosine
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --use-flash-attn
  --recompute-activations
  --save "${SAVE_DIR}"
  --load "${LOAD_CHECKPOINT}"
  --save-interval "${SAVE_INTERVAL:-500}"
  --log-interval "${LOG_INTERVAL:-10}"
)

if [[ -n "${VALID_DATA}" ]]; then
  cmd+=(--uniss-packed-valid "${VALID_DATA}" --eval-iters "${EVAL_ITERS:-10}" --eval-interval "${EVAL_INTERVAL:-500}")
else
  cmd+=(--eval-iters 0)
fi

cmd+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
