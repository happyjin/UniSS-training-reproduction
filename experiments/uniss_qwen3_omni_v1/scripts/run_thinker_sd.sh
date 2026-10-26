#!/usr/bin/env bash
# Plan 4A: the Thinker's self-distilled commitment (training/pretrain_thinker_sd.py) -- LoRA on the attention of the
# top text layers, adapter-only saves -- then merged into a full Qwen3-Omni copy vLLM serves (training/merge_thinker_sd.py).
#
#   RUN_ID=thinker_sd_$(date -u +%Y%m%dT%H%M%SZ) ROWS=data/si_thinker_selfdistill_v1/rows_train.jsonl \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
#   TRAIN_SEED / SD_SEED (opt-in) change the LoRA initialisation / data order seeds for replicate runs.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

RUN_ID="${RUN_ID:?set RUN_ID}"; ROWS="$(realpath "${ROWS:?rows_train.jsonl}")"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
MERGED="${MERGED:-${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/${RUN_ID}/thinker_merged}"
NPROC="${NPROC:-8}"; MBS="${MBS:-2}"; GBS="${GBS:-64}"
TRAIN_ITERS="${TRAIN_ITERS:-3000}"; RANK="${LORA_RANK:-32}"; ALPHA="${LORA_ALPHA:-64}"; FROM="${LORA_FROM:-32}"
if [ -e "${MERGED}" ] || [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "拒绝覆盖已有输出: ${MERGED} / ${SAVE_DIR}" >&2; exit 1
fi

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29811}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_thinker_sd.py
  --qwen3-path "${REPO_ROOT}/pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
  --sd-rows "${ROWS}" --sd-lora-rank "${RANK}" --sd-lora-alpha "${ALPHA}" --sd-lora-from "${FROM}"
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR:-1e-4}" --min-lr "${MIN_LR:-1e-5}" --lr-warmup-iters "${WARMUP_ITERS:-100}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding --num-workers "${NUM_WORKERS:-6}"
  --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-create-attention-mask-in-dataloader --no-gradient-accumulation-fusion
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}" --no-save-optim --no-save-rng
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  ${EXIT_INTERVAL:+--exit-interval "${EXIT_INTERVAL}"}
  --seed "${TRAIN_SEED:-20261022}" ${SD_SEED:+--sd-seed "${SD_SEED}"})

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "SAVE_DIR=${SAVE_DIR}  MERGED=${MERGED}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "THINKER_SD_FAILED status=${status}"; exit "${status}"; }
STEP=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
CUDA_VISIBLE_DEVICES="" "${ENV_ROOT}/bin/python" experiments/uniss_qwen3_omni_v1/training/merge_thinker_sd.py \
  --checkpoint-dir "${SAVE_DIR}/iter_$(printf %07d "${STEP}")" --rank "${RANK}" --alpha "${ALPHA}" --out "${MERGED}" \
  || { echo "THINKER_SD_MERGE_FAILED"; exit 1; }
echo "THINKER_SD_DONE ${MERGED}"
