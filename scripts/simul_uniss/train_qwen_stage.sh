#!/usr/bin/env bash
set -euo pipefail

STAGE=""
DRY_RUN=0
SMOKE=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${STAGE}" != "action" && "${STAGE}" != "interleaved" && "${STAGE}" != "joint" ]]; then
  echo "--stage must be action, interleaved, or joint" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

configure_python_nvidia_libraries() {
  local library_dirs=()
  local cuda_root directory site_packages joined
  if command -v nvcc >/dev/null 2>&1; then
    cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
    for directory in \
      "${cuda_root}/lib" \
      "${cuda_root}/lib64" \
      "${cuda_root}/targets/x86_64-linux/lib"; do
      [[ -d "${directory}" ]] && library_dirs+=("${directory}")
    done
  fi
  site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
  shopt -s nullglob
  for directory in "${site_packages}"/nvidia/*/lib; do
    [[ -d "${directory}" ]] && library_dirs+=("${directory}")
  done
  shopt -u nullglob
  (( ${#library_dirs[@]} > 0 )) || {
    echo "No CUDA or pip NVIDIA library directories found" >&2
    return 1
  }
  joined="$(IFS=:; echo "${library_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

if [[ "${DRY_RUN}" == "0" ]]; then
  configure_python_nvidia_libraries
  python - <<'PY'
import ctypes

ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402
PY
fi

if [[ "${STAGE}" == "action" ]]; then
  TRAIN_DATA="${ACTION_PACKED_TRAIN}"
  LOAD_ROOT="${STAGE3_LOAD_ROOT}"
  SAVE_ROOT="${STAGE3_SAVE_ROOT}"
  MASTER_PORT="${STAGE3_MASTER_PORT}"
  TRAIN_ITERS="${STAGE3_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE3_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_ACTION}"
  STAGE_WARMUP_ITERS="${STAGE3_QWEN_WARMUP_ITERS:-${SIMUL_QWEN_WARMUP_ITERS}}"
elif [[ "${STAGE}" == "interleaved" ]]; then
  TRAIN_DATA="${PACKED_TRAIN}"
  LOAD_ROOT="${STAGE4_LOAD_ROOT}"
  SAVE_ROOT="${STAGE4_SAVE_ROOT}"
  MASTER_PORT="${STAGE4_MASTER_PORT}"
  TRAIN_ITERS="${STAGE4_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE4_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_INTERLEAVED}"
  STAGE_WARMUP_ITERS="${STAGE4_QWEN_WARMUP_ITERS:-${SIMUL_QWEN_WARMUP_ITERS}}"
else
  TRAIN_DATA="${PACKED_TRAIN}"
  LOAD_ROOT="${STAGE6_LOAD_ROOT}"
  SAVE_ROOT="${STAGE6_SAVE_ROOT}"
  MASTER_PORT="${STAGE6_MASTER_PORT}"
  TRAIN_ITERS="${STAGE6_TRAIN_ITERS}"
  STAGE_TENSORBOARD_DIR="${STAGE6_TENSORBOARD_DIR}"
  VALID_DATA="${VALID_PACKED_INTERLEAVED}"
  STAGE_WARMUP_ITERS="${STAGE6_QWEN_WARMUP_ITERS:-${SIMUL_QWEN_WARMUP_ITERS}}"
fi

NPROC="${SIMUL_NPROC_PER_NODE}"
MICRO_BATCH="${SIMUL_MICRO_BATCH_SIZE}"
GLOBAL_BATCH="${SIMUL_GLOBAL_BATCH_SIZE}"
DATALOADER_TYPE="${SIMUL_DATALOADER_TYPE}"
NO_DATA_SHARDING="${SIMUL_NO_DATA_SHARDING:-0}"
FULL_VALIDATION="${SIMUL_FULL_VALIDATION:-0}"
OFFSET_INDEX_MODE="${SIMUL_OFFSET_INDEX_MODE:-sidecar}"
NO_LOAD_OPTIM="${SIMUL_NO_LOAD_OPTIM:-1}"
NO_LOAD_RNG="${SIMUL_NO_LOAD_RNG:-1}"
FINETUNE="${SIMUL_FINETUNE:-1}"
WARMUP_ITERS="${STAGE_WARMUP_ITERS}"
SAVE_INTERVAL="${SIMUL_QWEN_SAVE_INTERVAL}"
EVAL_INTERVAL="${SIMUL_QWEN_EVAL_INTERVAL}"
EVAL_ITERS="${SIMUL_QWEN_EVAL_ITERS}"
LOG_INTERVAL="${SIMUL_QWEN_LOG_INTERVAL:-1}"
TENSORBOARD_LOG_INTERVAL="${SIMUL_QWEN_TENSORBOARD_LOG_INTERVAL:-1}"
TENSORBOARD_MEMORY_INTERVAL="${SIMUL_QWEN_TENSORBOARD_MEMORY_INTERVAL:-1}"
QWEN_LR="${SIMUL_QWEN_LR}"
QWEN_MIN_LR="${SIMUL_QWEN_MIN_LR}"
if [[ "${STAGE}" == "joint" ]]; then
  QWEN_LR="${STAGE6_QWEN_LR}"
  QWEN_MIN_LR="${STAGE6_QWEN_MIN_LR}"
fi
if [[ "${NO_DATA_SHARDING}" != "0" && "${NO_DATA_SHARDING}" != "1" ]]; then
  echo "SIMUL_NO_DATA_SHARDING must be 0 or 1" >&2
  exit 2
fi
if [[ "${FULL_VALIDATION}" != "0" && "${FULL_VALIDATION}" != "1" ]]; then
  echo "SIMUL_FULL_VALIDATION must be 0 or 1" >&2
  exit 2
fi
if [[ "${OFFSET_INDEX_MODE}" != "sidecar" && "${OFFSET_INDEX_MODE}" != "phase3-scan" ]]; then
  echo "SIMUL_OFFSET_INDEX_MODE must be sidecar or phase3-scan" >&2
  exit 2
fi
for flag_name in NO_LOAD_OPTIM NO_LOAD_RNG FINETUNE; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "SIMUL_${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${SMOKE}" == "1" ]]; then
  NPROC=1
  MICRO_BATCH=1
  GLOBAL_BATCH=1
  TRAIN_ITERS=2
  WARMUP_ITERS=0
  SAVE_INTERVAL=1
  EVAL_INTERVAL=1
  EVAL_ITERS=1
  LOG_INTERVAL=1
  TENSORBOARD_LOG_INTERVAL=1
  TENSORBOARD_MEMORY_INTERVAL=1
fi

if [[ "${DRY_RUN}" == "0" && ! -f "${TRAIN_DATA}" ]]; then
  echo "Missing training data: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "0" && ! -f "${LOAD_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing load checkpoint pointer: ${LOAD_ROOT}/latest_checkpointed_iteration.txt" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "0" && ! -f "${VALID_DATA}" ]]; then
  echo "Missing validation data: ${VALID_DATA}" >&2
  exit 1
fi

cmd=(torchrun
  --nproc_per_node "${NPROC}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/training/pretrain_simul_uniss_megatron.py"
  --sft
  --simul-packed-train "${TRAIN_DATA}"
  --simul-schema-version simul_uniss_packed_v1
  --simul-offset-index-mode "${OFFSET_INDEX_MODE}"
  --tokenizer-type NullTokenizer
  --vocab-size 180407
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers 24
  --hidden-size 896
  --ffn-hidden-size 4864
  --num-attention-heads 14
  --group-query-attention
  --num-query-groups 2
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --add-qkv-bias
  --position-embedding-type rope
  --rotary-base 1000000
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings 32768
  --micro-batch-size "${MICRO_BATCH}"
  --global-batch-size "${GLOBAL_BATCH}"
  --dataloader-type "${DATALOADER_TYPE}"
  --seed "${SIMUL_DATA_SEED}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${QWEN_LR}"
  --min-lr "${QWEN_MIN_LR}"
  --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_ITERS}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --save "${SAVE_ROOT}"
  --load "${LOAD_ROOT}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --simul-packed-valid "${VALID_DATA}"
  --eval-iters "${EVAL_ITERS}"
  --eval-interval "${EVAL_INTERVAL}"
  --tensorboard-dir "${STAGE_TENSORBOARD_DIR}"
  --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${TENSORBOARD_MEMORY_INTERVAL}"
  --log-throughput
)

if [[ "${NO_LOAD_OPTIM}" == "1" ]]; then
  cmd+=(--no-load-optim)
fi
if [[ "${NO_LOAD_RNG}" == "1" ]]; then
  cmd+=(--no-load-rng)
fi
if [[ "${FINETUNE}" == "1" ]]; then
  cmd+=(--finetune)
fi

# Keep these opt-in so historical experiment configs continue to reproduce
# their original sampler and validation behavior. New experiments can request
# full-dataset random sampling and stable full validation independently.
if [[ "${NO_DATA_SHARDING}" == "1" ]]; then
  cmd+=(--no-data-sharding)
fi
if [[ "${FULL_VALIDATION}" == "1" ]]; then
  cmd+=(--full-validation)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${SAVE_ROOT}" "${STAGE_TENSORBOARD_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage_${STAGE}_qwen.log"
