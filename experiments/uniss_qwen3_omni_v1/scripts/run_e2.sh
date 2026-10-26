#!/usr/bin/env bash
# E2: Qwen3-Omni's Thinker (frozen) with UniSS as its Talker, trained as one
# model on UniST S2ST -- the full 19.2M-row audio pool in a real permutation,
# reshuffled per pass. See training/e2_model.py.
#
# The base rate is for UniSS, which is already trained; --decoupled-lr is
# for the three added inputs, which start at zero (e2_model flags them).
# Only the trainable 1.5B is checkpointed, about 21 GB a save with its
# optimizer state, so SAVE_INTERVAL is set with the disk in mind.
#
#   RUN_ID=e2_$(date -u +%Y%m%dT%H%M%SZ) bash experiments/uniss_qwen3_omni_v1/scripts/run_e2.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
#   INIT_FROM=<another run's checkpoint dir> INIT_STEP=N starts this run from
#   that run's iter N (read only; this run saves to its own SAVE_DIR). Used
#   only while SAVE_DIR has no checkpoint; after that the run resumes itself.
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
export UNISS_RESHUFFLE_EACH_EPOCH=1

RUN_ID="${RUN_ID:?set RUN_ID, e.g. e2_20261020T190000Z}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
NPROC="${NPROC:-8}"; MBS="${MBS:-8}"; GBS="${GBS:-256}"
TRAIN_ITERS="${TRAIN_ITERS:-8000}"; WARMUP_ITERS="${WARMUP_ITERS:-200}"
LR="${LR:-1e-5}"; MIN_LR="${MIN_LR:-1e-6}"
ADDED_LR="${ADDED_LR:-5e-4}"; ADDED_MIN_LR="${ADDED_MIN_LR:-5e-5}"
LOAD_ARG=""
if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  LOAD_ARG="--load ${SAVE_DIR}"
elif [ -n "${INIT_FROM:-}" ]; then
  [ -d "${INIT_FROM}/iter_$(printf %07d "${INIT_STEP:?INIT_FROM needs INIT_STEP}")" ] \
    || { echo "没有 ${INIT_FROM} 的第 ${INIT_STEP} 步" >&2; exit 1; }
  LOAD_ARG="--load ${INIT_FROM} --ckpt-step ${INIT_STEP}"
fi

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29711}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_e2.py
  --uniss-path "${UNISS_PATH:-${REPO_ROOT}/pretrained_models/UniSS_full}"
  --omni-audio-index "${AUDIO_INDEX:-${REPO_ROOT}/data/unist_omni_v1/audio_index_full}"
  --omni-parquet-root "${REPO_ROOT}/data/raw/UniST"
  --omni-audio-root "${REPO_ROOT}/data/unist_omni_v1/speech"
  --omni-max-audio-seconds "${MAX_AUDIO_SECONDS:-30}"
  --e2-duration-dropout "${DURATION_DROPOUT:-0.2}"
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR}" --min-lr "${MIN_LR}"
  --decoupled-lr "${ADDED_LR}" --decoupled-min-lr "${ADDED_MIN_LR}"
  --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding
  --num-workers "${NUM_WORKERS:-6}"
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --use-distributed-optimizer
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}"
  ${LOAD_ARG}
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  ${EXIT_INTERVAL:+--exit-interval "${EXIT_INTERVAL}"}
  ${NO_ADDED:+--e2-no-added} --e2-kl-weight "${KL_WEIGHT:-0}" --e2-glm-source "${GLM_SOURCE:-released}"
  ${OVERRIDE_SCHED:-}
  --seed 20261020)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "SAVE_DIR=${SAVE_DIR}  LOAD=${LOAD_ARG:-none}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "E2_MEGATRON_FAILED status=${status}"; exit "${status}"; }
echo "E2_MEGATRON_DONE"
