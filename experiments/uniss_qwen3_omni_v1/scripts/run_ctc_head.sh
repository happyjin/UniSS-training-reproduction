#!/usr/bin/env bash
# Plan 2's CTC head (BiCodec semantic codes -> characters) under Megatron (training/pretrain_ctc_head.py),
# on rows from si_ctc_head.py prep, then exported as a head si_ctc_head.load_ctc reads.
#
#   RUN_ID=ctc_head_$(date -u +%Y%m%dT%H%M%SZ) ROWS=data/si_ctc_v1/rows.npz \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_ctc_head.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
# PASSES over the training rows (default 8); BUDGET padded codes a bucket (a micro-batch is one
# bucket); the step count follows from the rows and NPROC.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:?set RUN_ID, e.g. ctc_head_20261022T090000Z}"
ROWS="$(realpath "${ROWS:?the prep npz}")"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
OUT="${OUT:-${REPO_ROOT}/data/si_ctc_v1/heads/${RUN_ID}.pt}"
NPROC="${NPROC:-8}"; PASSES="${PASSES:-8}"; BUDGET="${BUDGET:-48000}"
D="${D:-512}"; LAYERS="${LAYERS:-8}"; HEADS="${HEADS:-8}"; FF="${FF:-2048}"
GBS="${NPROC}"
if [ -e "${OUT}" ] || [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "拒绝覆盖已有输出: ${OUT} / ${SAVE_DIR}" >&2; exit 1
fi
# buckets a pass ~ padded frames / budget; padding inside length-sorted windows is small, count it at 5%
TRAIN_FRAMES=$("${ENV_ROOT}/bin/python" -c "import numpy as np; z=np.load('${ROWS}'); l=np.diff(z['code_off']); print(int(l[~z['val']].sum()))")
TRAIN_ITERS="${TRAIN_ITERS:-$(( PASSES * TRAIN_FRAMES * 105 / 100 / BUDGET / GBS ))}"

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29791}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_ctc_head.py
  --ctc-rows "${ROWS}" --ctc-frame-budget "${BUDGET}"
  --ctc-d "${D}" --ctc-layers "${LAYERS}" --ctc-heads "${HEADS}" --ctc-ff "${FF}"
  --tokenizer-type NullTokenizer --vocab-size 64
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 1 --hidden-size 64 --ffn-hidden-size 256 --num-attention-heads 1
  --seq-length 32 --max-position-embeddings 32
  --micro-batch-size 1 --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR:-1e-3}" --min-lr "${MIN_LR:-5e-5}" --lr-warmup-iters "${WARMUP_ITERS:-1000}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-6}"
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0
  --bf16 --no-gradient-accumulation-fusion
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-${TRAIN_ITERS}}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-50}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  --seed 20261022)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "train frames ${TRAIN_FRAMES}  train iters ${TRAIN_ITERS}  SAVE_DIR=${SAVE_DIR}  OUT=${OUT}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}" "$(dirname "${OUT}")"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "CTC_HEAD_FAILED status=${status}"; exit "${status}"; }
ITER_DIR="${SAVE_DIR}/iter_$(printf %07d "${TRAIN_ITERS}")"
"${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/export_ctc_head.py \
  --checkpoint-dir "${ITER_DIR}" --rows "${ROWS}" --out "${OUT}" \
  --dims "{\"d\": ${D}, \"layers\": ${LAYERS}, \"heads\": ${HEADS}, \"ff\": ${FF}}" \
  || { echo "CTC_HEAD_EXPORT_FAILED"; exit 1; }
echo "CTC_HEAD_DONE ${OUT}"
