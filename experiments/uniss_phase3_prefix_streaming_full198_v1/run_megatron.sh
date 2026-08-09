#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env"

SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
[[ -x "${PYTHON}" ]] || { echo "Missing experiment Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${PHASE3_MODEL}" ]] || { echo "Missing Phase3 model: ${PHASE3_MODEL}" >&2; exit 1; }
[[ -f "${INDEX_JSON}" ]] || { echo "Missing full198 direction index: ${INDEX_JSON}" >&2; exit 1; }
[[ -f "${VALID_PARQUET}" ]] || { echo "Missing UniST dev parquet: ${VALID_PARQUET}" >&2; exit 1; }

if [[ "${SMOKE}" != "1" && "${NPROC_PER_NODE}" != "8" ]]; then
  echo "Formal run requires 8 GPUs" >&2
  exit 1
fi
if [[ "${SMOKE}" != "1" && "${GLOBAL_BATCH_SIZE}" != "128" ]]; then
  echo "Formal run requires global batch 128" >&2
  exit 1
fi

if [[ "${RESUME}" == "1" ]]; then
  [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]] || {
    echo "RESUME=1 but no checkpoint tracker exists in ${SAVE_DIR}" >&2
    exit 1
  }
  LOAD_ARGS=(--load "${SAVE_DIR}")
else
  for path in "${SAVE_DIR}" "${TB_DIR}" "${LOG}"; do
    [[ ! -e "${path}" ]] || { echo "Refusing to overwrite existing experiment path: ${path}" >&2; exit 1; }
  done
  LOAD_ARGS=()
fi

mkdir -p "$(dirname "${LOG}")" "$(dirname "${TB_DIR}")" "$(dirname "${SAVE_DIR}")"

SITE_PACKAGES="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
LIB_DIRS=()
shopt -s nullglob
for directory in "${SITE_PACKAGES}"/nvidia/*/lib; do
  [[ -d "${directory}" ]] && LIB_DIRS+=("${directory}")
done
shopt -u nullglob
if (( ${#LIB_DIRS[@]} > 0 )); then
  NVIDIA_LIBRARY_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

EXPERIMENT_ARGS=()
[[ "${SMOKE}" == "1" ]] && EXPERIMENT_ARGS+=(--experiment-smoke)

export CUDA_VISIBLE_DEVICES
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/trainer.py" \
  --experiment-index-json "${INDEX_JSON}" \
  --experiment-valid-parquet "${VALID_PARQUET}" \
  --experiment-phase3-model "${PHASE3_MODEL}" \
  --experiment-valid-limit "${VALID_LIMIT}" \
  --experiment-block-size 4096 \
  --experiment-cache-shards 2 \
  --experiment-lora-rank 16 \
  --experiment-lora-alpha 32 \
  --experiment-lora-dropout 0.05 \
  --experiment-lora-targets q_proj,v_proj \
  --experiment-teacher-topk 32 \
  --experiment-teacher-temperature 1.5 \
  --experiment-confidence-threshold 0.70 \
  --experiment-min-write-tokens 2 \
  --experiment-history-tokens 200 \
  --experiment-max-sample-tokens 4096 \
  --experiment-teacher-kl-weight 0.25 \
  --experiment-semantic-kl-weight 0.20 \
  --experiment-consistency-weight 0.20 \
  --experiment-commit-consistency-weight 0.25 \
  --experiment-boundary-weight 0.10 \
  --experiment-action-weight 1.0 \
  --experiment-attention-implementation flash_attention_2 \
  "${EXPERIMENT_ARGS[@]}" \
  --tokenizer-type NullTokenizer \
  --vocab-size 180407 \
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
  --bf16 \
  --seq-length 18000 \
  --max-position-embeddings 32768 \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --lr-warmup-iters "${WARMUP_ITERS}" \
  --lr-decay-iters "${TRAIN_ITERS}" \
  --lr-decay-style cosine \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --clip-grad 0.5 \
  --dataloader-type single \
  --num-workers "${NUM_WORKERS}" \
  --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-iters "${EVAL_ITERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --tensorboard-dir "${TB_DIR}" \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --log-memory-to-tensorboard \
  --seed "${SEED}" \
  "${LOAD_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"

