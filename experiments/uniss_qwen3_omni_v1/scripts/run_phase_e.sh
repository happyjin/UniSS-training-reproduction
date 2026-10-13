#!/usr/bin/env bash
# Phase E: the Phase D Talker learns to read Qwen3's S2TT reply states.
#
# SAVE_INTERVAL is 2000 and only the last few checkpoints are kept. Phase
# D wrote 14 of them at 65 GB each -- 904 GB -- because Megatron stores
# the frozen 31.7B Thinker in every one, identical every time, while the
# 393M that actually train export to 300 MB. The cost of a save belongs in
# the configuration, not in a cleanup afterwards.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

PHASE_D="${PHASE_D:?set PHASE_D to a phase_d export}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/phase_e}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/phase_e}"
DEV="${DEV:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_dev_v1}"
SUBSET="${SUBSET:-${REPO_ROOT}/data/cvss_t/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl}"
mkdir -p "${SAVE_DIR}" "${TB_DIR}"

NPROC="${NPROC:-8}"; MBS="${MBS:-8}"; GBS="${GBS:-256}"
TRAIN_ITERS="${TRAIN_ITERS:-15000}"; WARMUP_ITERS="${WARMUP_ITERS:-200}"
LR="${LR:-6e-5}"; MIN_LR="${MIN_LR:-2e-6}"
EXTRA=()
[ "${NO_RECOMPUTE:-0}" = "1" ] && EXTRA+=(--omni-no-recompute)
LOAD_ARG=""
[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && LOAD_ARG="--load ${SAVE_DIR}"

"${ENV_ROOT}/bin/python" -m torch.distributed.run \
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29701}" \
  experiments/uniss_qwen3_omni_v1/training/pretrain_phase_e.py \
  --phase-d-checkpoint "${PHASE_D}" \
  --omni-audio-index "${AUDIO_INDEX:-${REPO_ROOT}/data/unist_omni_v1/audio_index_full}" \
  --omni-parquet-root "${REPO_ROOT}/data/raw/UniST" \
  --omni-audio-root "${REPO_ROOT}/data/unist_omni_v1/speech" \
  --omni-tts-cache "${REPO_ROOT}/data/unist_omni_v1/tts_cache_full" \
  --omni-dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
  --omni-dev-subset "${SUBSET}" \
  --omni-audio-limit "${AUDIO_LIMIT:-0}" \
  --omni-max-audio-seconds "${MAX_AUDIO_SECONDS:-30}" \
  --omni-head-scale "${HEAD_SCALE:-0.1}" \
  --tokenizer-type NullTokenizer --vocab-size 8448 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
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
  --num-workers "${NUM_WORKERS:-6}" \
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
  --bf16 --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-2000}" \
  ${LOAD_ARG} \
  --eval-interval "${EVAL_INTERVAL:-500}" --eval-iters "${EVAL_ITERS:-4}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --tensorboard-dir "${TB_DIR}" --log-throughput \
  --seed 20261013 "${EXTRA[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "PHASE_E_MEGATRON_FAILED status=${status}"; exit "${status}"; }
echo "PHASE_E_MEGATRON_DONE"
