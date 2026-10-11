#!/usr/bin/env bash
# Stage 2 joint pretraining, 8 GPUs, Megatron.
#
# Trainable scope is the paper's: Thinker on LoRA r32, Talker full, audio
# encoder frozen. LR 1e-4 -> 2e-5, also the paper's.
#
# The GPT geometry flags describe the Talker and exist because Megatron's
# config container requires them; model_provider returns the Omni wrapper.
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

DEV="${DEV:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_dev_v1}"
SUBSET="${SUBSET:-${REPO_ROOT}/data/cvss_t/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage2}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_omni_s2st_v1/stage2}"
TALKER="${TALKER:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt}"

mkdir -p "${SAVE_DIR}" "${TB_DIR}"
EXTRA=()
[[ -n "${EXIT_INTERVAL:-}" ]] && EXTRA+=(--exit-interval "${EXIT_INTERVAL}")
if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  EXTRA+=(--load "${SAVE_DIR}")
  echo "resuming from iteration $(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
fi

"${ENV_ROOT}/bin/torchrun" \
  --nproc_per_node "${NPROC:-8}" --master_port "${MASTER_PORT:-29601}" \
  experiments/uniss_omni_s2st_v1/training/pretrain_stage2_megatron.py \
  --omni-model pretrained_models/Qwen2.5-Omni-3B \
  --omni-talker-state "${TALKER}" \
  --omni-manifests "${REPO_ROOT}/data/unist_omni_v1/manifests/manifest-train-*.jsonl" \
  --omni-parquet-root "${REPO_ROOT}/data/raw/UniST" \
  --omni-audio-root "${REPO_ROOT}/data/unist_omni_v1/speech" \
  --omni-tts-cache "${REPO_ROOT}/data/unist_omni_v1/tts_cache_full" \
  --omni-dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
  --omni-dev-subset "${SUBSET}" \
  --omni-audio-limit "${AUDIO_LIMIT:-0}" \
  --omni-lora-rank "${LORA_RANK:-32}" \
  --omni-lora-alpha "${LORA_ALPHA:-64}" \
  --omni-code-weight "${CODE_WEIGHT:-1.0}" \
  --omni-max-audio-seconds "${MAX_AUDIO_SECONDS:-30}" \
  --tokenizer-type NullTokenizer --vocab-size 8448 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864 \
  --num-attention-heads 14 --group-query-attention --num-query-groups 2 \
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias \
  --position-embedding-type rope --rotary-base 1000000 \
  --seq-length 1024 --max-position-embeddings 32768 \
  --micro-batch-size "${MBS:-8}" --global-batch-size "${GBS:-64}" \
  --train-iters "${TRAIN_ITERS:-8000}" \
  --lr "${LR:-1e-4}" --min-lr "${MIN_LR:-2e-5}" \
  --lr-warmup-iters "${WARMUP_ITERS:-200}" \
  --lr-decay-iters "${TRAIN_ITERS:-8000}" --lr-decay-style cosine \
  --dataloader-type single --no-data-sharding \
  --num-workers "${NUM_WORKERS:-6}" \
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
  --bf16 --no-create-attention-mask-in-dataloader --no-gradient-accumulation-fusion \
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" --eval-iters "${EVAL_ITERS:-8}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --tensorboard-dir "${TB_DIR}" --log-throughput --seed "${SEED:-20261011}" \
  "${EXTRA[@]}"
echo "STAGE2 MEGATRON DONE"
