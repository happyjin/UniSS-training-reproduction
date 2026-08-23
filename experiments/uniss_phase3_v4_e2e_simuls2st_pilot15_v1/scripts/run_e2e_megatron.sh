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
: "${RUN_TRAIN_BUILD_REPORT:?RUN_TRAIN_BUILD_REPORT is required}"
: "${RUN_SAVE_DIR:?RUN_SAVE_DIR is required}"
: "${RUN_TENSORBOARD_DIR:?RUN_TENSORBOARD_DIR is required}"
: "${RUN_LOG:?RUN_LOG is required}"

RUN_VALID_BUILD_REPORT=${RUN_VALID_BUILD_REPORT:-}
RUN_V1_TRAIN_CACHE_AUDIT=${RUN_V1_TRAIN_CACHE_AUDIT:-}
RUN_PHASE3_TRAIN_CACHE_AUDIT=${RUN_PHASE3_TRAIN_CACHE_AUDIT:-}
RUN_V1_VALID_CACHE_AUDIT=${RUN_V1_VALID_CACHE_AUDIT:-}
RUN_PHASE3_VALID_CACHE_AUDIT=${RUN_PHASE3_VALID_CACHE_AUDIT:-}
RUN_TRAINING_GATE=${RUN_TRAINING_GATE:-}
RUN_LOAD=${RUN_LOAD:-$(dirname "${V1_CHECKPOINT}")}
RUN_NPROC=${RUN_NPROC:-8}
RUN_MBS=${RUN_MBS:-1}
RUN_GBS=${RUN_GBS:-128}
RUN_COVERAGE_EPOCHS=${RUN_COVERAGE_EPOCHS:-3}
RUN_NUM_WORKERS=${RUN_NUM_WORKERS:-8}
RUN_MASTER_PORT=${RUN_MASTER_PORT:-29731}
RUN_SAVE_INTERVAL=${RUN_SAVE_INTERVAL:-50}
RUN_EVAL_INTERVAL=${RUN_EVAL_INTERVAL:-50}
RUN_EVAL_ITERS=${RUN_EVAL_ITERS:-0}
RUN_LOG_INTERVAL=${RUN_LOG_INTERVAL:-5}
RUN_SEED=${RUN_SEED:-20260818}
RUN_SMOKE=${RUN_SMOKE:-0}
RUN_SMOKE_FAMILY=${RUN_SMOKE_FAMILY:-}
RUN_LEARNING_CANARY=${RUN_LEARNING_CANARY:-0}
RUN_PHASE_STRATIFIED_CANARY=${RUN_PHASE_STRATIFIED_CANARY:-0}
RUN_CANARY_REPORT=${RUN_CANARY_REPORT:-}
RUN_ALLOW_MISSING_TEACHERS=${RUN_ALLOW_MISSING_TEACHERS:-0}
RUN_AUDIT_GRADIENTS=${RUN_AUDIT_GRADIENTS:-0}
RUN_VERIFY_DATASET_SHA256=${RUN_VERIFY_DATASET_SHA256:-0}
RUN_VERIFY_CACHE_SHA256=${RUN_VERIFY_CACHE_SHA256:-0}
RUN_CONTENT_END_WEIGHT=${RUN_CONTENT_END_WEIGHT:-0.0}
RUN_SEMANTIC_END_WEIGHT=${RUN_SEMANTIC_END_WEIGHT:-0.0}
RUN_SEMANTIC_END_MARGIN_WEIGHT=${RUN_SEMANTIC_END_MARGIN_WEIGHT:-0.0}
RUN_SEMANTIC_END_LOGIT_MARGIN=${RUN_SEMANTIC_END_LOGIT_MARGIN:-0.0}
RUN_SEMANTIC_PREFIX_CORRUPTION_RATE=${RUN_SEMANTIC_PREFIX_CORRUPTION_RATE:-0.0}
RUN_SEMANTIC_PREFIX_CORRUPTION_TAIL=${RUN_SEMANTIC_PREFIX_CORRUPTION_TAIL:-8}
RUN_SEMANTIC_PREFIX_CORRUPTION_RAMP_UPDATES=${RUN_SEMANTIC_PREFIX_CORRUPTION_RAMP_UPDATES:-0}
RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE=${RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE:-0.0}
RUN_SEMANTIC_BOUNDARY_ROLLIN_RAMP_UPDATES=${RUN_SEMANTIC_BOUNDARY_ROLLIN_RAMP_UPDATES:-0}

[[ "${RUN_PHASE_STRATIFIED_CANARY}" == "0" || "${RUN_PHASE_STRATIFIED_CANARY}" == "1" ]] || {
  echo "RUN_PHASE_STRATIFIED_CANARY must be zero or one" >&2
  exit 4
}

ENTRYPOINT="${EXPERIMENT_DIR}/training/pretrain_e2e_megatron.py"
FINGERPRINTS="${PROCESSED_ROOT}/manifests/checkpoint_fingerprints.json"
GEOMETRY=${RUN_GEOMETRY:-${REPORT_ROOT}/training_geometry/${RUN_ID}.json}
GEOMETRY_DIR=$(dirname "${GEOMETRY}")
V1_STATIC_AUDIT=${RUN_V1_STATIC_AUDIT:-${GEOMETRY_DIR}/${RUN_ID}.v1_checkpoint_audit.json}

export HF_HOME="${USER_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${USER_ROOT}/.cache/pip"
export TMPDIR="${USER_ROOT}/tmp"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
NVIDIA_LIBRARY_ROOT="$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia"
NVIDIA_LIBRARY_PATH=
if [[ -d "${NVIDIA_LIBRARY_ROOT}" ]]; then
  NVIDIA_LIBRARY_PATH=$(find "${NVIDIA_LIBRARY_ROOT}" \
    -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
fi
# This host provides the Transformer-Engine-compatible cuDNN 9.7 under the
# CUDA toolkit.  Keep system CUDA libraries ahead of the pip fallback: putting
# the environment's older cuDNN 9.1 first imports successfully but fails on the
# first fused-attention call with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
SYSTEM_CUDA_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${SYSTEM_CUDA_LIBRARY_PATH}:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}${NVIDIA_LIBRARY_PATH:+:${NVIDIA_LIBRARY_PATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UNISS_E2E_COMPILE_CACHE_ROOT="${USER_ROOT}/.cache/uniss_e2e_compile/${RUN_ID}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${UNISS_E2E_COMPILE_CACHE_ROOT}"

required=(
  "${RUN_TRAIN_BUILD_REPORT}"
  "${RUN_LOAD}/latest_checkpointed_iteration.txt"
  "${WHISPERVQ_MODEL}/config.json"
  "${FINGERPRINTS}"
)
for value in "${required[@]}"; do
  [[ -f "${value}" ]] || { echo "missing E2E input: ${value}" >&2; exit 1; }
done

if [[ "${RUN_ALLOW_MISSING_TEACHERS}" != "1" ]]; then
  for value in "${RUN_V1_TRAIN_CACHE_AUDIT}" "${RUN_PHASE3_TRAIN_CACHE_AUDIT}"; do
    [[ -f "${value}" ]] || { echo "missing E2E train teacher cache: ${value}" >&2; exit 1; }
  done
fi
if [[ -n "${RUN_VALID_BUILD_REPORT}" ]]; then
  [[ -f "${RUN_VALID_BUILD_REPORT}" ]] || { echo "missing E2E valid report" >&2; exit 1; }
  if [[ "${RUN_ALLOW_MISSING_TEACHERS}" != "1" ]]; then
    for value in "${RUN_V1_VALID_CACHE_AUDIT}" "${RUN_PHASE3_VALID_CACHE_AUDIT}"; do
      [[ -f "${value}" ]] || { echo "missing E2E valid teacher cache: ${value}" >&2; exit 1; }
    done
  fi
fi
if [[ "${RUN_SMOKE}" != "1" && "${RUN_LEARNING_CANARY}" != "1" ]]; then
  [[ -f "${RUN_TRAINING_GATE}" ]] || { echo "missing formal E2E training gate" >&2; exit 1; }
fi
if [[ "${RUN_LEARNING_CANARY}" == "1" ]]; then
  [[ -f "${RUN_CANARY_REPORT}" ]] || { echo "missing passed structural canary report" >&2; exit 1; }
fi

if [[ ! -f "${GEOMETRY}" ]]; then
  mkdir -p "${GEOMETRY_DIR}"
  "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.compute_geometry \
    --task-pool-report "${RUN_TRAIN_BUILD_REPORT}" \
    --global-batch-size "${RUN_GBS}" \
    --coverage-epochs "${RUN_COVERAGE_EPOCHS}" \
    --seed "${RUN_SEED}" \
    --output "${GEOMETRY}" >/dev/null
fi
if [[ ! -f "${V1_STATIC_AUDIT}" ]]; then
  mkdir -p "$(dirname "${V1_STATIC_AUDIT}")"
  "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.audit_v1_checkpoint \
    --v1-load-root "${RUN_LOAD}" \
    --phase3-checkpoint "${PHASE3_CHECKPOINT}" \
    --fingerprints "${FINGERPRINTS}" \
    --output "${V1_STATIC_AUDIT}" >/dev/null
fi
RUN_TRAIN_ITERS=${RUN_TRAIN_ITERS:-$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_iters"])' "${GEOMETRY}")}
if [[ -z "${RUN_WARMUP_ITERS+x}" ]]; then
  if [[ "${RUN_SMOKE}" == "1" ]]; then
    RUN_WARMUP_ITERS=0
  else
    RUN_WARMUP_ITERS=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["warmup_updates"])' "${GEOMETRY}")
  fi
fi

if [[ "${RUN_ALLOW_MISSING_TEACHERS}" == "1" && "${RUN_SMOKE}" != "1" ]]; then
  echo "missing E2E teachers are allowed only in smoke mode" >&2
  exit 4
fi
if [[ "${RUN_SMOKE}" == "1" && ( "${RUN_TRAIN_ITERS}" -lt 1 || "${RUN_TRAIN_ITERS}" -gt 2 ) ]]; then
  echo "E2E smoke runs are restricted to one or two updates" >&2
  exit 4
fi
if [[ "${RUN_LEARNING_CANARY}" == "1" && ( "${RUN_TRAIN_ITERS}" -lt 10 || "${RUN_TRAIN_ITERS}" -gt 100 ) ]]; then
  echo "E2E learning canaries are restricted to 10--100 updates" >&2
  exit 4
fi
if [[ "${RUN_SMOKE}" == "1" && "${RUN_LEARNING_CANARY}" == "1" ]]; then
  echo "E2E smoke and learning canary are mutually exclusive" >&2
  exit 4
fi
if [[ "${RUN_PHASE_STRATIFIED_CANARY}" == "1" && "${RUN_LEARNING_CANARY}" != "1" ]]; then
  echo "phase-stratified E2E canary requires learning-canary mode" >&2
  exit 4
fi
if [[ "${RUN_WARMUP_ITERS}" -lt 0 || "${RUN_WARMUP_ITERS}" -gt "${RUN_TRAIN_ITERS}" ]]; then
  echo "E2E warmup updates must be between zero and train-iters" >&2
  exit 4
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${RUN_SAVE_DIR}" || -e "${RUN_TENSORBOARD_DIR}" || -e "${RUN_LOG}" ]]; then
    echo "refusing to overwrite E2E run ${RUN_ID}" >&2
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
  --e2e-train-build-report "${RUN_TRAIN_BUILD_REPORT}"
  --e2e-whispervq-model "${WHISPERVQ_MODEL}"
  --e2e-checkpoint-fingerprints "${FINGERPRINTS}"
  --e2e-coverage-epochs "${RUN_COVERAGE_EPOCHS}"
  --e2e-lr-qwen 2e-6
  --e2e-lr-qwen-io 5e-7
  --e2e-content-end-weight "${RUN_CONTENT_END_WEIGHT}"
  --e2e-semantic-end-weight "${RUN_SEMANTIC_END_WEIGHT}"
  --e2e-semantic-end-margin-weight "${RUN_SEMANTIC_END_MARGIN_WEIGHT}"
  --e2e-semantic-end-logit-margin "${RUN_SEMANTIC_END_LOGIT_MARGIN}"
  --e2e-semantic-prefix-corruption-rate "${RUN_SEMANTIC_PREFIX_CORRUPTION_RATE}"
  --e2e-semantic-prefix-corruption-tail "${RUN_SEMANTIC_PREFIX_CORRUPTION_TAIL}"
  --e2e-semantic-prefix-corruption-ramp-updates "${RUN_SEMANTIC_PREFIX_CORRUPTION_RAMP_UPDATES}"
  --e2e-semantic-boundary-rollin-rate "${RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE}"
  --e2e-semantic-boundary-rollin-ramp-updates "${RUN_SEMANTIC_BOUNDARY_ROLLIN_RAMP_UPDATES}"
  --e2e-speaker-continuity-weight 0.0
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
  --seq-length 18000
  --max-position-embeddings 32768
  --micro-batch-size "${RUN_MBS}"
  --global-batch-size "${RUN_GBS}"
  --train-iters "${RUN_TRAIN_ITERS}"
  --lr 2e-6
  --min-lr 2e-7
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
  --dist-ckpt-strictness raise_unexpected
  --finetune
  --no-load-optim
  --no-load-rng
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
  --seed "${RUN_SEED}"
)

[[ -n "${RUN_V1_TRAIN_CACHE_AUDIT}" ]] && cmd+=(--e2e-v1-train-cache-audit "${RUN_V1_TRAIN_CACHE_AUDIT}")
[[ -n "${RUN_PHASE3_TRAIN_CACHE_AUDIT}" ]] && cmd+=(--e2e-phase3-train-cache-audit "${RUN_PHASE3_TRAIN_CACHE_AUDIT}")
[[ "${RUN_SMOKE}" == "1" ]] && cmd+=(--e2e-smoke)
[[ -n "${RUN_SMOKE_FAMILY}" ]] && cmd+=(--e2e-smoke-family "${RUN_SMOKE_FAMILY}")
[[ "${RUN_LEARNING_CANARY}" == "1" ]] && cmd+=(--e2e-learning-canary --e2e-canary-report "${RUN_CANARY_REPORT}")
[[ "${RUN_PHASE_STRATIFIED_CANARY}" == "1" ]] && cmd+=(--e2e-phase-stratified-canary)
[[ "${RUN_ALLOW_MISSING_TEACHERS}" == "1" ]] && cmd+=(--e2e-allow-missing-teachers)
[[ "${RUN_AUDIT_GRADIENTS}" == "1" ]] && cmd+=(--e2e-audit-gradients)
[[ "${RUN_VERIFY_DATASET_SHA256}" == "1" ]] && cmd+=(--e2e-verify-dataset-sha256)
[[ "${RUN_VERIFY_CACHE_SHA256}" == "1" ]] && cmd+=(--e2e-verify-cache-sha256)
[[ -n "${RUN_TRAINING_GATE}" ]] && cmd+=(--e2e-training-gate "${RUN_TRAINING_GATE}")

if [[ -n "${RUN_VALID_BUILD_REPORT}" ]]; then
  cmd+=(
    --e2e-valid-build-report "${RUN_VALID_BUILD_REPORT}"
    --e2e-v1-valid-cache-audit "${RUN_V1_VALID_CACHE_AUDIT}"
    --e2e-phase3-valid-cache-audit "${RUN_PHASE3_VALID_CACHE_AUDIT}"
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
cp -- "${GEOMETRY}" "${RUN_LOG}.geometry.json"
cp -- "${V1_STATIC_AUDIT}" "${RUN_LOG}.v1_checkpoint_audit.json"

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
