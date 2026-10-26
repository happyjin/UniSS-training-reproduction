#!/usr/bin/env bash
# Plan 2's amortised self-monitoring under Megatron (training/pretrain_selfmon.py): the talker, from T1c@1500, fit to
# the continuations the Thinker chose (si_infer --select-k --dump-seq rows), weights-only saves, then exported.
#
#   RUN_ID=selfmon_$(date -u +%Y%m%dT%H%M%SZ) ROWS='data/si_selfmon_v1/*/part_*.jsonl' \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_selfmon.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

T1C="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
RUN_ID="${RUN_ID:?set RUN_ID}"; ROWS="${ROWS:?glob of dump-seq results}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
EXPORT_OUT="${EXPORT_OUT:-${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/${RUN_ID}/export}"
NPROC="${NPROC:-8}"; GBS="${NPROC}"; TRAIN_ITERS="${TRAIN_ITERS:-400}"
if [ -e "${EXPORT_OUT}" ] || [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "拒绝覆盖已有输出: ${EXPORT_OUT} / ${SAVE_DIR}" >&2; exit 1
fi

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29801}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_selfmon.py
  --uniss-path "${UNISS_PATH:-${T1C}}" --ref-path "${REF_PATH:-${T1C}}"
  --selfmon-rows "${ROWS}" --selfmon-w-ref "${W_REF:-0.5}" --selfmon-close-weight "${CLOSE_WEIGHT:-3}"
  --selfmon-token-budget "${BUDGET:-24000}"
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size 1 --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR:-3e-6}" --min-lr "${MIN_LR:-1e-6}" --lr-warmup-iters "${WARMUP_ITERS:-20}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-4}"
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer
  --save "${SAVE_DIR}" --save-interval "${TRAIN_ITERS}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  --seed 20261022)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "SAVE_DIR=${SAVE_DIR}  EXPORT_OUT=${EXPORT_OUT}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "SELFMON_FAILED status=${status}"; exit "${status}"; }
CUDA_VISIBLE_DEVICES="" "${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/export_e2.py \
  --checkpoint-dir "${SAVE_DIR}/iter_$(printf %07d "${TRAIN_ITERS}")" --output "${EXPORT_OUT}" \
  || { echo "SELFMON_EXPORT_FAILED"; exit 1; }
echo "SELFMON_DONE ${EXPORT_OUT}"
