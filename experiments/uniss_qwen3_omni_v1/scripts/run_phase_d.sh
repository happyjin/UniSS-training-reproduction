#!/usr/bin/env bash
# Phase D: the Stage 1 Talker learns to read Qwen3-Omni's hidden states.
#
# The GPT geometry below describes the Talker (24 layers, 896 hidden, 4864
# FFN, 14 heads over 2 KV groups) because Megatron's config container
# requires it. It does not build the model -- model_provider assembles
# Qwen3's Thinker with Qwen2.5's Talker -- but keeping it truthful means
# the logged config describes what actually trains.
#
# ENV_ROOT is the qwen3omni overlay, not uniss-train: Qwen3-Omni needs
# transformers 5.x and the overlay leaves the base environments untouched.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# inductor shells out to triton's ptxas, which lives in the base
# environment and is not executable here, so every compiled region raises
# PermissionError and takes the run with it. Nothing in this model needs
# compiling -- the Thinker runs under no_grad and the Talker is 24 small
# layers -- so dynamo is off and everything runs eager.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

CACHE="${CACHE:-${REPO_ROOT}/data/unist_omni_v1/tts_cache_full}"
TALKER="${TALKER:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/phase_d}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/phase_d}"
mkdir -p "${SAVE_DIR}" "${TB_DIR}"

NPROC="${NPROC:-8}"
MBS="${MBS:-4}"
GBS="${GBS:-384}"
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
WARMUP_ITERS="${WARMUP_ITERS:-200}"
# Stage 1 peaked at 1.258e-4 and annealed to 3.5e-6 over 40,000 steps. The
# Talker here starts trained rather than transferred, so the peak is lower;
# the floor is kept proportionally low, which stage 2 did not do and paid
# 1.1 to 1.8 BLEU for.
LR="${LR:-6e-5}"
MIN_LR="${MIN_LR:-2e-6}"
EXTRA=()
[ "${NO_RECOMPUTE:-0}" = "1" ] && EXTRA+=(--omni-no-recompute)

# python -m torch.distributed.run rather than the torchrun console script:
# the overlay venv inherits torch from the base environment but has no
# entry point of its own, and the base environment's torchrun would run
# the base python, which is the transformers 4.53 this cannot use.
"${ENV_ROOT}/bin/python" -m torch.distributed.run \
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29601}" \
  experiments/uniss_qwen3_omni_v1/training/pretrain_phase_d.py \
  --qwen3-path "${QWEN3:-pretrained_models/Qwen3-Omni-30B-A3B-Instruct}" \
  --qwen25-path "${QWEN25:-pretrained_models/Qwen2.5-Omni-3B}" \
  --talker-checkpoint "${TALKER}" \
  --omni-cache "${CACHE}" \
  --omni-dev-rows "${DEV_ROWS:-4096}" \
  --omni-max-text-tokens "${MAX_TEXT_TOKENS:-256}" \
  --omni-head-scale "${HEAD_SCALE:-0.1}" \
  --tokenizer-type NullTokenizer \
  --vocab-size 8448 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864 \
  --num-attention-heads 14 --group-query-attention --num-query-groups 2 \
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias \
  --position-embedding-type rope --rotary-base 1000000 \
  --seq-length 640 --max-position-embeddings 32768 \
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr "${LR}" --min-lr "${MIN_LR}" \
  --lr-warmup-iters "${WARMUP_ITERS}" \
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine \
  --dataloader-type single --no-data-sharding \
  --num-workers "${NUM_WORKERS:-4}" \
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
  --bf16 \
  --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-500}" --eval-iters "${EVAL_ITERS:-8}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --tensorboard-dir "${TB_DIR}" --log-throughput \
  --seed 20261012 \
  "${EXTRA[@]}"
status=$?
# Without this a failed launch still prints the completion marker, and
# anything waiting on that marker treats a crash as a finished run.
[ "${status}" -ne 0 ] && { echo "PHASE_D_MEGATRON_FAILED status=${status}"; exit "${status}"; }
echo "PHASE_D_MEGATRON_DONE"
