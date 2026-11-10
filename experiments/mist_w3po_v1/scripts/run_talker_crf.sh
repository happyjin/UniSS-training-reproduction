#!/usr/bin/env bash
# Talker schemes (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §0): PL-EM (MODE=plem) or the alignment-CRF talker
# (MODE=align) under Megatron (experiments/mist_w3po_v1/talker/pretrain_talker_crf.py), from T1c's export, full talker,
# LR 3e-5 (warmup 50, cosine to 3e-6), 1200 steps, 4 cards (CARDS), one row group per card per step (GBS 4), KL 0.5 to
# the frozen T1c, weights-only saves every 400 steps, then exported like T1c (export_e2.py; align: the head in added.pt).
# Checked with bash -n / DRY_RUN=1 only; never test-run.
#   RUN_ID=talker_plem_$(date -u +%Y%m%dT%H%M%SZ) MODE=plem ROWS=<rows.pt> CARDS=0,1,2,3 MASTER_PORT=29811 \
#     setsid nohup bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

T1C="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
RUN_ID="${RUN_ID:?set RUN_ID}"; MODE="${MODE:?plem|align}"; ROWS="${ROWS:?rows file}"; CARDS="${CARDS:?e.g. 0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CARDS}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/mist_w3po_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/mist_w3po_v1/${RUN_ID}}"
EXPORT_OUT="${EXPORT_OUT:-${REPO_ROOT}/eval_outputs/mist_w3po_v1/talker_crf/${RUN_ID}/export}"
NPROC=$(echo "${CARDS}" | tr ',' '\n' | grep -c .); GBS="${NPROC}"; TRAIN_ITERS="${TRAIN_ITERS:-1200}"
if [ -e "${EXPORT_OUT}" ] || [ -e "${SAVE_DIR}" ]; then
  echo "拒绝覆盖已有输出: ${EXPORT_OUT} / ${SAVE_DIR}" >&2; exit 1
fi
PFX=""; [ "${MODE}" = align ] && PFX="--added-prefixes align_"

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29811}"
  experiments/mist_w3po_v1/talker/pretrain_talker_crf.py
  --crf-mode "${MODE}" --uniss-path "${UNISS_PATH:-${T1C}}" --ref-path "${REF_PATH:-${T1C}}"
  --crf-rows "${ROWS}" --crf-w-ref 0.5 --crf-w-speed 1.0 --crf-w-align 1.0 --crf-w-dur 0.1 --crf-close-weight 3
  --crf-token-budget "${BUDGET:-24000}"
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size 1 --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr 3e-5 --min-lr 3e-6 --lr-warmup-iters 50
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-4}"
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-400}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  --seed 20261110)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SAVE_DIR=${SAVE_DIR} EXPORT_OUT=${EXPORT_OUT}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "TALKER_CRF_FAILED status=${status}"; exit "${status}"; }
CUDA_VISIBLE_DEVICES="" "${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/export_e2.py \
  --checkpoint-dir "${SAVE_DIR}/iter_$(printf %07d "${TRAIN_ITERS}")" --base "${T1C}" --output "${EXPORT_OUT}" ${PFX} \
  || { echo "TALKER_CRF_EXPORT_FAILED"; exit 1; }
echo "TALKER_CRF_DONE ${EXPORT_OUT}"
