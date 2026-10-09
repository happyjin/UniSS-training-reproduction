#!/usr/bin/env bash
# Stage 1 Talker warmup, on Megatron, 8 GPUs.
#
# The GPT geometry flags below describe the Talker (24 layers, 896 hidden,
# 4864 FFN, 14 heads, 2 KV groups) and exist because Megatron's config
# container requires them. They do not build the model -- model_provider
# returns Qwen2.5-Omni wrapped in Megatron's HuggingFaceModule -- but
# keeping them truthful means the logged config describes what is training.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

DEV=/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_dev_v1
SUBSET=/opt/dlami/nvme/jasonleeeli/CVSS/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1_megatron}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_omni_s2st_v1/stage1_megatron}"

NPROC="${NPROC:-8}"
MBS="${MBS:-16}"
GBS="${GBS:-128}"
TRAIN_ITERS="${TRAIN_ITERS:-15000}"
WARMUP_ITERS="${WARMUP_ITERS:-200}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-2e-6}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-0}"
EXTRA=()
[[ -n "${EXIT_INTERVAL:-}" ]] && EXTRA+=(--exit-interval "${EXIT_INTERVAL}")
[[ "${GLOBAL_PREFIX:-0}" == "1" ]] && EXTRA+=(--omni-global-prefix)
# Resume if this save directory already holds a checkpoint. Megatron then
# restores the optimiser, the LR schedule and consumed_train_samples, so an
# interrupted run continues rather than restarting from zero.
if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  EXTRA+=(--load "${SAVE_DIR}")
  echo "resuming from iteration $(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
fi

SHARDS=(train-00001.parquet train-00004.parquet train-00013.parquet train-00030.parquet
        train-00040.parquet train-00067.parquet train-00070.parquet train-00124.parquet
        train-00142.parquet train-00164.parquet)

# A prebuilt cache or the raw shards, never both. An array rather than a
# ${VAR:+...} expansion: the latter splits the path on whitespace and the
# failure surfaces as argparse rejecting a stray positional.
if [[ -n "${OMNI_CACHE:-}" ]]; then
  SOURCE_ARGS=(--omni-cache "${OMNI_CACHE}")
else
  SOURCE_ARGS=(--omni-shards "${SHARDS[@]}")
fi

mkdir -p "${SAVE_DIR}" "${TB_DIR}"

"${ENV_ROOT}/bin/torchrun" \
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29541}" \
  experiments/uniss_omni_s2st_v1/training/pretrain_omni_talker_megatron.py \
  --omni-model pretrained_models/Qwen2.5-Omni-3B \
  --omni-parquet-root data/raw/UniST \
  "${SOURCE_ARGS[@]}" \
  --omni-dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
  --omni-dev-subset "${SUBSET}" \
  --omni-rows-per-shard "${ROWS_PER_SHARD}" \
  --omni-head-scale "${HEAD_SCALE:-0.1}" \
  --omni-embed-scale "${EMBED_SCALE:-1.0}" \
  --tokenizer-type NullTokenizer \
  --vocab-size 8448 \
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
  --seq-length 640 \
  --max-position-embeddings 32768 \
  --micro-batch-size "${MBS}" \
  --global-batch-size "${GBS}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr "${LR}" --min-lr "${MIN_LR}" \
  --lr-warmup-iters "${WARMUP_ITERS}" \
  --lr-decay-iters "${TRAIN_ITERS}" \
  --lr-decay-style cosine \
  --dataloader-type single \
  --no-data-sharding \
  --num-workers "${NUM_WORKERS:-2}" \
  --weight-decay 0.01 \
  --adam-beta1 0.9 --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --bf16 \
  --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-250}" \
  --eval-iters "${EVAL_ITERS:-16}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --tensorboard-dir "${TB_DIR}" \
  --log-throughput \
  --seed 20261009 \
  "${EXTRA[@]}"
echo "STAGE1 MEGATRON DONE"
