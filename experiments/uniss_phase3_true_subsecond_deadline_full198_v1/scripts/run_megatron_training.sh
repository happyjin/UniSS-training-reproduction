#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_SAVE_DIR:?RUN_SAVE_DIR is required}"
: "${RUN_TB_DIR:?RUN_TB_DIR is required}"
: "${RUN_LOG:?RUN_LOG is required}"
: "${RUN_TRAJECTORY_PACKED:?RUN_TRAJECTORY_PACKED is required}"
: "${RUN_TRAJECTORY_OFFSETS:?RUN_TRAJECTORY_OFFSETS is required}"
: "${RUN_REPLAY_PACKED:?RUN_REPLAY_PACKED is required}"
: "${RUN_REPLAY_OFFSETS:?RUN_REPLAY_OFFSETS is required}"
: "${RUN_TRAIN_ITERS:?RUN_TRAIN_ITERS is required}"
: "${RUN_LOAD:?RUN_LOAD is required}"

RUN_NPROC="${RUN_NPROC:-8}"
RUN_MBS="${RUN_MBS:-2}"
RUN_GBS="${RUN_GBS:-128}"
RUN_MASTER_PORT="${RUN_MASTER_PORT:-${MASTER_PORT}}"
RUN_WARMUP_ITERS="${RUN_WARMUP_ITERS:-200}"
RUN_SAVE_INTERVAL="${RUN_SAVE_INTERVAL:-${SAVE_INTERVAL}}"
RUN_EVAL_INTERVAL="${RUN_EVAL_INTERVAL:-${EVAL_INTERVAL}}"
RUN_LOG_INTERVAL="${RUN_LOG_INTERVAL:-${LOG_INTERVAL}}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_LOAD_OPTIM="${RUN_LOAD_OPTIM:-0}"
RUN_LOAD_RNG="${RUN_LOAD_RNG:-0}"
RUN_STRICTNESS="${RUN_STRICTNESS:-log_all}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_EXIT_INTERVAL="${RUN_EXIT_INTERVAL:-}"
RUN_VALID_REPLAY_PACKED="${RUN_VALID_REPLAY_PACKED:-}"
RUN_VALID_REPLAY_OFFSETS="${RUN_VALID_REPLAY_OFFSETS:-}"
RUN_FULL_VALIDATION="${RUN_FULL_VALIDATION:-0}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${RUN_SAVE_DIR}" "${RUN_TB_DIR}" \
  "$(dirname "${RUN_LOG}")"

required=(
  "${RUN_TRAJECTORY_PACKED}" "${RUN_TRAJECTORY_OFFSETS}"
  "${RUN_REPLAY_PACKED}" "${RUN_REPLAY_OFFSETS}"
  "${WHISPERVQ_CODEBOOK}" "${PHASE3_FINGERPRINT}"
  "${RUN_LOAD}/latest_checkpointed_iteration.txt"
)
for value in "${required[@]}"; do
  [[ -f "${value}" ]] || { echo "Missing required file: ${value}" >&2; exit 1; }
done
if [[ "${DRY_RUN}" != "1" ]]; then
  visible_gpus="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
  [[ "${visible_gpus}" == "${RUN_NPROC}" ]] || {
    echo "Expected ${RUN_NPROC} visible GPUs, found ${visible_gpus}" >&2
    exit 1
  }
fi

cmd=(
  "${ENV_ROOT}/bin/torchrun"
  --nproc_per_node "${RUN_NPROC}"
  --master_port "${RUN_MASTER_PORT}"
  "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/training/pretrain_true_subsecond_megatron.py"
  --sft
  --true-trajectory-packed "${RUN_TRAJECTORY_PACKED}"
  --true-trajectory-offsets "${RUN_TRAJECTORY_OFFSETS}"
  --true-replay-packed "${RUN_REPLAY_PACKED}"
  --true-replay-offsets "${RUN_REPLAY_OFFSETS}"
  --true-whispervq-codebook "${WHISPERVQ_CODEBOOK}"
  --true-phase3-fingerprint "${PHASE3_FINGERPRINT}"
  --true-lora-rank 32
  --true-lora-alpha 64
  --true-lora-dropout 0.05
  --true-lora-mlp-last-layers 12
  --true-lr-qwen-lora "${LR_QWEN_LORA}"
  --true-lr-frontend "${LR_FRONTEND}"
  --true-lr-new-heads "${LR_NEW_HEADS}"
  --true-min-lr "${MIN_LR}"
  --true-npz-lru-capacity "${NPZ_LRU_CAPACITY}"
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
  --seq-length 18000
  --max-position-embeddings 32768
  --micro-batch-size "${RUN_MBS}"
  --global-batch-size "${RUN_GBS}"
  --train-iters "${RUN_TRAIN_ITERS}"
  --lr "${LR_NEW_HEADS}"
  --min-lr "${MIN_LR}"
  --lr-warmup-iters "${RUN_WARMUP_ITERS}"
  --lr-decay-iters "${RUN_TRAIN_ITERS}"
  --lr-decay-style cosine
  --dataloader-type cyclic
  --no-data-sharding
  --num-workers "${NUM_WORKERS}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 0.5
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --check-for-nan-in-loss-and-grad
  --dist-ckpt-strictness "${RUN_STRICTNESS}"
  --save "${RUN_SAVE_DIR}"
  --load "${RUN_LOAD}"
  --save-interval "${RUN_SAVE_INTERVAL}"
  --log-interval "${RUN_LOG_INTERVAL}"
  --tensorboard-dir "${RUN_TB_DIR}"
  --tensorboard-log-interval "${RUN_LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${RUN_LOG_INTERVAL}"
  --log-world-size-to-tensorboard
  --log-throughput
  --seed "${SEED}"
)
[[ "${RUN_FINETUNE}" == "1" ]] && cmd+=(--finetune)
[[ "${RUN_LOAD_OPTIM}" != "1" ]] && cmd+=(--no-load-optim)
[[ "${RUN_LOAD_RNG}" != "1" ]] && cmd+=(--no-load-rng)
[[ "${RUN_SMOKE}" == "1" ]] && cmd+=(--true-smoke --true-allow-partial-index)
[[ -n "${RUN_EXIT_INTERVAL}" ]] && cmd+=(--exit-interval "${RUN_EXIT_INTERVAL}")
if [[ -n "${RUN_VALID_REPLAY_PACKED}" ]]; then
  [[ -f "${RUN_VALID_REPLAY_PACKED}" && -f "${RUN_VALID_REPLAY_OFFSETS}" ]] || {
    echo "Validation replay data/index is incomplete" >&2
    exit 1
  }
  cmd+=(
    --true-valid-replay-packed "${RUN_VALID_REPLAY_PACKED}"
    --true-valid-replay-offsets "${RUN_VALID_REPLAY_OFFSETS}"
    --eval-iters "${EVAL_ITERS}"
    --eval-interval "${RUN_EVAL_INTERVAL}"
  )
  [[ "${RUN_FULL_VALIDATION}" == "1" ]] && cmd+=(
    --full-validation --eval-micro-batch-size 1 --eval-global-batch-size "${RUN_NPROC}"
  )
else
  cmd+=(--eval-iters 0)
fi
cmd+=("$@")

printf '%q ' "${cmd[@]}" > "${RUN_LOG}.command"
printf '\n' >> "${RUN_LOG}.command"
if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${RUN_LOG}.command"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee -a "${RUN_LOG}"
