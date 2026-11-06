#!/usr/bin/env bash
# W3PO stage-0 SFT (plan §4): run_thinker_gstar.sh with the S0 entry (rows may carry "ctx"); outputs under
# eval_outputs/mist_w3po_v1/{ckpt,models}/RUN_ID, logs/mist_w3po_v1/tb. v8 recipe: BASE=O2 merged, INIT_MERGED=1, 3000 iters.
#   RUN_ID=... ROWS=... BASE=... INIT_MERGED=1 TRAIN_ITERS=3000 setsid nohup bash run_s0_sft.sh > log 2>&1 < /dev/null &
# DRY_RUN=1 prints the command; SKIP_MERGE=1 stops after training (benchmarks).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
J=/opt/dlami/nvme/jasonleeeli
export HF_HOME="${HF_HOME:-$J/cache/huggingface}" TMPDIR="$J/work/mist_w3po_v1/tmp" TORCH_HOME="$J/cache/torch" XDG_CACHE_HOME="$J/cache" TRITON_CACHE_DIR="$J/cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

RUN_ID="${RUN_ID:?set RUN_ID}"; ROWS="$(realpath "${ROWS:?rows_train.jsonl}")"
BASE="$(realpath "${BASE:-pretrained_models/Qwen3-Omni-30B-A3B-Instruct}")"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/eval_outputs/mist_w3po_v1/ckpt/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/logs/mist_w3po_v1/tb/${RUN_ID}}"
MERGED="${MERGED:-${REPO_ROOT}/eval_outputs/mist_w3po_v1/models/${RUN_ID}/thinker_merged}"
NPROC="${NPROC:-8}"; MBS="${MBS:-8}"; GBS="${GBS:-64}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"; RANK="${LORA_RANK:-32}"; ALPHA="${LORA_ALPHA:-64}"; FROM="${LORA_FROM:-0}"
if [ -e "${MERGED}" ] || [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "拒绝覆盖已有输出: ${MERGED} / ${SAVE_DIR}" >&2; exit 1
fi

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29815}"
  experiments/mist_w3po_v1/training/pretrain_thinker_gstar_s0.py
  --qwen3-path "${BASE}"
  --gstar-rows "${ROWS}" --gstar-lora-rank "${RANK}" --gstar-lora-alpha "${ALPHA}" --gstar-lora-from "${FROM}"
  ${NOSD:+--gstar-no-sd} ${ORACLE:+--gstar-oracle} ${INIT_MERGED:+--gstar-init-merged}
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR:-1e-4}" --min-lr "${MIN_LR:-1e-5}" --lr-warmup-iters "${WARMUP_ITERS:-50}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-8}"
  --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-create-attention-mask-in-dataloader --no-gradient-accumulation-fusion
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-250}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  ${EXIT_INTERVAL:+--exit-interval "${EXIT_INTERVAL}"}
  --seed "${TRAIN_SEED:-20261026}" --gstar-seed "${GSTAR_SEED:-20261026}")

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "SAVE_DIR=${SAVE_DIR}  MERGED=${MERGED}  BASE=${BASE}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "THINKER_GSTAR_FAILED status=${status}"; exit "${status}"; }
[ -n "${SKIP_MERGE:-}" ] && { echo "THINKER_GSTAR_TRAINED (no merge)"; exit 0; }
STEP=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
CUDA_VISIBLE_DEVICES="" "${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/merge_thinker_gstar.py \
  --checkpoint-dir "${SAVE_DIR}/iter_$(printf %07d "${STEP}")" --rank "${RANK}" --alpha "${ALPHA}" \
  --base "${BASE}" --out "${MERGED}" || { echo "THINKER_GSTAR_MERGE_FAILED"; exit 1; }
echo "THINKER_GSTAR_DONE ${MERGED}"
