#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?RUN_ID is required}"
: "${RUN_TRAIN_PACKS:?RUN_TRAIN_PACKS is required}"
: "${RUN_SAVE_DIR:?RUN_SAVE_DIR is required}"
: "${RUN_TENSORBOARD_DIR:?RUN_TENSORBOARD_DIR is required}"
: "${RUN_LOG:?RUN_LOG is required}"

RUN_VALID_PACKS=${RUN_VALID_PACKS:-}
RUN_LOAD=${RUN_LOAD:-${PHASE3_NATIVE_ROOT}}
RUN_NPROC=${RUN_NPROC:-8}
RUN_SEQ_LENGTH=${RUN_SEQ_LENGTH:-18000}
RUN_MBS=${RUN_MBS:-1}
RUN_GBS=${RUN_GBS:-128}
RUN_COVERAGE_EPOCHS=${RUN_COVERAGE_EPOCHS:-3}
RUN_TRAIN_ITERS=${RUN_TRAIN_ITERS:?RUN_TRAIN_ITERS is required}
RUN_MAX_ACOUSTICS=${RUN_MAX_ACOUSTICS:-2}
RUN_NUM_WORKERS=${RUN_NUM_WORKERS:-4}
RUN_MASTER_PORT=${RUN_MASTER_PORT:-29671}
RUN_SAVE_INTERVAL=${RUN_SAVE_INTERVAL:-100}
RUN_EVAL_INTERVAL=${RUN_EVAL_INTERVAL:-100}
RUN_EVAL_ITERS=${RUN_EVAL_ITERS:-0}
RUN_LOG_INTERVAL=${RUN_LOG_INTERVAL:-10}
RUN_WARMUP_ITERS=${RUN_WARMUP_ITERS:-200}
RUN_STRICTNESS=${RUN_STRICTNESS:-log_all}
RUN_SMOKE=${RUN_SMOKE:-0}
RUN_AUDIT_GRADIENTS=${RUN_AUDIT_GRADIENTS:-0}
RUN_FINETUNE=${RUN_FINETUNE:-1}
RUN_LOAD_OPTIM=${RUN_LOAD_OPTIM:-0}
RUN_LOAD_RNG=${RUN_LOAD_RNG:-0}

PHASE3_FINGERPRINT="${REPO_ROOT}/data/processed/uniss_phase3_true_subsecond_deadline_full198_v1/model_handoff/phase3_embedding_fingerprint.json"
FRONTEND_GATE="${REPORT_ROOT}/stage_a_causal_whisper_asr/frontend_parity/20260816T204500Z/FRONTEND_TRAINING_GATE_PASSED.json"
ENTRYPOINT="${EXPERIMENT_DIR}/stage_a_causal_whisper_asr/training/pretrain_stage_a_megatron.py"

export HF_HOME="${JASON_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${JASON_ROOT}/.cache/pip"
export TMPDIR="${JASON_ROOT}/tmp"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UNISS_STAGE_A_COMPILE_CACHE_ROOT="${JASON_ROOT}/.cache/stage_a_compile/${RUN_ID}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${UNISS_STAGE_A_COMPILE_CACHE_ROOT}"

required=(
  "${RUN_TRAIN_PACKS}"
  "${RUN_LOAD}/latest_checkpointed_iteration.txt"
  "${WHISPERVQ_CHECKPOINT}/config.json"
  "${PHASE3_FINGERPRINT}"
  "${FRONTEND_GATE}"
)
for value in "${required[@]}"; do
  [[ -f "${value}" ]] || { echo "missing Stage A input: ${value}" >&2; exit 1; }
done
if [[ -n "${RUN_VALID_PACKS}" && ! -f "${RUN_VALID_PACKS}" ]]; then
  echo "missing Stage A validation packs: ${RUN_VALID_PACKS}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${RUN_SAVE_DIR}" || -e "${RUN_TENSORBOARD_DIR}" || -e "${RUN_LOG}" ]]; then
    echo "refusing to overwrite Stage A run ${RUN_ID}" >&2
    exit 2
  fi
  mkdir -p "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}" "$(dirname "${RUN_LOG}")"
  visible=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
  [[ "${visible}" == "${RUN_NPROC}" ]] || {
    echo "expected ${RUN_NPROC} visible GPUs, found ${visible}" >&2
    exit 3
  }
  "${PYTHON_BIN}" -c 'import transformer_engine.pytorch' >/dev/null
fi

cmd=(
  "$(dirname "${PYTHON_BIN}")/torchrun"
  --nproc_per_node "${RUN_NPROC}"
  --master_port "${RUN_MASTER_PORT}"
  "${ENTRYPOINT}"
  --sft
  --stage-a-train-packs "${RUN_TRAIN_PACKS}"
  --stage-a-whispervq-model "${WHISPERVQ_CHECKPOINT}"
  --stage-a-frontend-gate "${FRONTEND_GATE}"
  --stage-a-phase3-fingerprint "${PHASE3_FINGERPRINT}"
  --stage-a-coverage-epochs "${RUN_COVERAGE_EPOCHS}"
  --stage-a-max-acoustics-per-pack "${RUN_MAX_ACOUSTICS}"
  --stage-a-lr-new-head 1e-4
  --stage-a-lr-bridge 5e-5
  --stage-a-lr-whisper-top 1e-6
  --stage-a-lr-whisper-bottom 2e-7
  --stage-a-lr-whisper-conv 1e-7
  --stage-a-lr-qwen 2e-6
  --stage-a-lr-qwen-io 5e-7
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
  --seq-length "${RUN_SEQ_LENGTH}"
  --max-position-embeddings 32768
  --micro-batch-size "${RUN_MBS}"
  --global-batch-size "${RUN_GBS}"
  --train-iters "${RUN_TRAIN_ITERS}"
  --lr 1e-4
  --min-lr 1e-5
  --lr-warmup-iters "${RUN_WARMUP_ITERS}"
  --lr-decay-iters "${RUN_TRAIN_ITERS}"
  --lr-decay-style cosine
  --dataloader-type cyclic
  --no-data-sharding
  --num-workers "${RUN_NUM_WORKERS}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 0.5
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --attention-dropout 0.1
  --hidden-dropout 0.1
  --dist-ckpt-strictness "${RUN_STRICTNESS}"
  --save "${RUN_SAVE_DIR}"
  --load "${RUN_LOAD}"
  --save-interval "${RUN_SAVE_INTERVAL}"
  --log-interval "${RUN_LOG_INTERVAL}"
  --tensorboard-dir "${RUN_TENSORBOARD_DIR}"
  --tensorboard-log-interval "${RUN_LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${RUN_LOG_INTERVAL}"
  --log-world-size-to-tensorboard
  --log-throughput
  --seed 20260816
)

[[ "${RUN_FINETUNE}" == "1" ]] && cmd+=(--finetune)
[[ "${RUN_LOAD_OPTIM}" != "1" ]] && cmd+=(--no-load-optim)
[[ "${RUN_LOAD_RNG}" != "1" ]] && cmd+=(--no-load-rng)
[[ "${RUN_SMOKE}" == "1" ]] && cmd+=(--stage-a-smoke)
[[ "${RUN_AUDIT_GRADIENTS}" == "1" ]] && cmd+=(--stage-a-audit-gradients)

if [[ -n "${RUN_VALID_PACKS}" ]]; then
  cmd+=(
    --stage-a-valid-packs "${RUN_VALID_PACKS}"
    --eval-iters "${RUN_EVAL_ITERS}"
    --eval-interval "${RUN_EVAL_INTERVAL}"
    --eval-micro-batch-size 1
    --eval-global-batch-size "${RUN_NPROC}"
  )
else
  cmd+=(--eval-iters 0 --eval-interval "${RUN_EVAL_INTERVAL}")
fi
cmd+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf '%q ' "${cmd[@]}" > "${RUN_LOG}.command"
printf '\n' >> "${RUN_LOG}.command"

MONITOR_LOG="${RUN_LOG%.log}.gpu.csv"
(
  echo "timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w,power_limit_w"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "${MONITOR_LOG}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
"${cmd[@]}" 2>&1 | tee "${RUN_LOG}"
