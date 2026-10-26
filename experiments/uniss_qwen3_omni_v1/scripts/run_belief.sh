#!/usr/bin/env bash
# The commit belief under Megatron (training/pretrain_belief.py) on trajectories from
# si_belief.py prep, then exported as the checkpoint si_thinker_stream --policy belief loads.
#
#   RUN_ID=belief_$(date -u +%Y%m%dT%H%M%SZ) ROWS=data/si_belief_v1/rows.npz \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_belief.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
# EPOCHS passes over the training trajectories (default 6); the step count follows
# from them, CHUNK trajectories a sample, MBS samples a micro-batch, NPROC data-parallel ranks.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

RUN_ID="${RUN_ID:?set RUN_ID, e.g. belief_20261022T180000Z}"
ROWS="$(realpath "${ROWS:?the prep npz}")"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
OUT="${OUT:-${REPO_ROOT}/data/si_belief_v1/models/${RUN_ID}.pt}"
NPROC="${NPROC:-4}"; MBS="${MBS:-8}"; CHUNK="${CHUNK:-512}"; EPOCHS="${EPOCHS:-6}"
GBS=$(( MBS * NPROC ))
if [ -e "${OUT}" ] || [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "拒绝覆盖已有输出: ${OUT} / ${SAVE_DIR}" >&2; exit 1
fi
TRAIN_ROWS=$("${ENV_ROOT}/bin/python" -c "import numpy as np; print(int((~np.load('${ROWS}')['val']).sum()))")
TRAIN_ITERS="${TRAIN_ITERS:-$(( EPOCHS * (TRAIN_ROWS / CHUNK) / GBS ))}"

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29791}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_belief.py
  --belief-rows "${ROWS}" --belief-chunk "${CHUNK}"
  --tokenizer-type NullTokenizer --vocab-size 64
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 1 --hidden-size 64 --ffn-hidden-size 256 --num-attention-heads 1
  --seq-length 32 --max-position-embeddings 32
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR:-2e-3}" --min-lr "${MIN_LR:-2e-4}" --lr-warmup-iters "${WARMUP_ITERS:-50}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-4}"
  --weight-decay 1e-4 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  --no-gradient-accumulation-fusion
  --save "${SAVE_DIR}" --save-interval "${TRAIN_ITERS}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-50}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  --seed 20261022)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "train rows ${TRAIN_ROWS}  train iters ${TRAIN_ITERS}  SAVE_DIR=${SAVE_DIR}  OUT=${OUT}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "BELIEF_FAILED status=${status}"; exit "${status}"; }
ITER_DIR="${SAVE_DIR}/iter_$(printf %07d "${TRAIN_ITERS}")"
CUDA_VISIBLE_DEVICES="" "${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/export_belief.py \
  --checkpoint-dir "${ITER_DIR}" --rows "${ROWS}" --out "${OUT}" --check \
  || { echo "BELIEF_EXPORT_FAILED"; exit 1; }
echo "BELIEF_DONE ${OUT}"
