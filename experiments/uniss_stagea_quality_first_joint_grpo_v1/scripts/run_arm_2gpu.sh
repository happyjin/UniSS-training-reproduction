#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {a1_sft|a2_g4|a3_g8|a4_g8_seed2} [--smoke]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
ARM=$1
SMOKE=${2:-}

case "${ARM}" in
  a1_sft)
    MODE=sft; GROUP=4; GPUS=0,1; PORT=29941; SEED=20260825; ANCHOR=0.01; SFT_WEIGHT=1.0 ;;
  a2_g4)
    MODE=grpo; GROUP=4; GPUS=2,3; PORT=29942; SEED=20260825; ANCHOR=0.01; SFT_WEIGHT=0.20 ;;
  a3_g8)
    MODE=grpo; GROUP=8; GPUS=4,5; PORT=29943; SEED=20260825; ANCHOR=0.01; SFT_WEIGHT=0.20 ;;
  a4_g8_seed2)
    MODE=grpo; GROUP=8; GPUS=6,7; PORT=29944; SEED=20260925; ANCHOR=0.02; SFT_WEIGHT=0.30 ;;
  *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac

TRAIN_ITERS_LOCAL=${TRAIN_ITERS}
BOOTSTRAP_LOCAL=${BOOTSTRAP_UPDATES}
EVAL_ITERS_LOCAL=${EVAL_ITERS}
SUFFIX=full
SMOKE_ARGS=()
WARMUP_ITERS=50
if [[ "${SMOKE}" == "--smoke" ]]; then
  TRAIN_ITERS_LOCAL=2
  BOOTSTRAP_LOCAL=1
  EVAL_ITERS_LOCAL=1
  SUFFIX="smoke_$(date -u +%Y%m%dT%H%M%SZ)"
  WARMUP_ITERS=0
  SMOKE_ARGS+=(--joint-smoke)
elif [[ -n "${SMOKE}" ]]; then
  echo "unknown option: ${SMOKE}" >&2
  exit 2
fi

RUN_VARIANT=${RUN_VARIANT:-}
if [[ -n "${RUN_VARIANT}" && "${SMOKE}" != "--smoke" ]]; then
  SUFFIX=${RUN_VARIANT}
fi
DATA_WORKERS=${DATA_WORKERS:-0}
if ! [[ "${DATA_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "DATA_WORKERS must be a non-negative integer" >&2
  exit 2
fi
RUN_ID=${ARM}_${SUFFIX}
SAVE_DIR=${CHECKPOINT_ROOT}/${RUN_ID}
TB_DIR=${RUN_ROOT}/tensorboard/${RUN_ID}
LOG=${LOG_ROOT}/${RUN_ID}.log
[[ ! -e "${SAVE_DIR}" && ! -e "${TB_DIR}" && ! -e "${LOG}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2
  exit 3
}
mkdir -p "${CHECKPOINT_ROOT}" "${TB_DIR}" "${LOG_ROOT}" "${REPORT_ROOT}" "${EVAL_ROOT}"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PIP_CACHE_DIR=${USER_ROOT}/.cache/pip
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON_BIN}"):${PATH}
NVIDIA_LIBRARY_ROOT=$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia
NVIDIA_LIBRARY_PATH=
if [[ -d "${NVIDIA_LIBRARY_ROOT}" ]]; then
  NVIDIA_LIBRARY_PATH=$(find "${NVIDIA_LIBRARY_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
fi
SYSTEM_CUDA_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=${SYSTEM_CUDA_LIBRARY_PATH}:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}${NVIDIA_LIBRARY_PATH:+:${NVIDIA_LIBRARY_PATH}}
export CUDA_VISIBLE_DEVICES=${GPUS}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export UNISS_E2E_COMPILE_CACHE_ROOT=${USER_ROOT}/.cache/uniss_stagea_joint_grpo/${RUN_ID}
mkdir -p "${HF_HOME}" "${TMPDIR}" "${UNISS_E2E_COMPILE_CACHE_ROOT}"

ENTRYPOINT=${EXPERIMENT_ROOT}/training/pretrain_megatron.py
CMD=(
  $(dirname "${PYTHON_BIN}")/torchrun
  --nproc_per_node 2
  --master_port "${PORT}"
  "${ENTRYPOINT}"
  --sft
  --joint-mode "${MODE}"
  --joint-group-size "${GROUP}"
  --joint-bootstrap-updates "${BOOTSTRAP_LOCAL}"
  --joint-candidate-width 16
  --joint-clip-epsilon 0.20
  --joint-kl-beta 0.02
  --joint-sft-replay-weight "${SFT_WEIGHT}"
  --joint-reference-anchor-weight "${ANCHOR}"
  --joint-lora-rank 16
  --joint-lora-alpha 32
  --joint-lora-dropout 0.05
  --joint-top-layers 8
  --joint-adapter-lr "${ADAPTER_LR}"
  "${SMOKE_ARGS[@]}"
  --e2e-train-build-report "${TRAIN_REPORT}"
  --e2e-valid-build-report "${VALID_REPORT}"
  --e2e-phase3-train-cache-audit "${PHASE3_TRAIN_CACHE}"
  --e2e-phase3-valid-cache-audit "${PHASE3_VALID_CACHE}"
  --e2e-whispervq-model "${WHISPERVQ_MODEL}"
  --e2e-checkpoint-fingerprints "${FINGERPRINTS}"
  --e2e-asr-weight 1.0
  --e2e-mt-weight 1.0
  --e2e-semantic-weight 1.0
  --e2e-replay-weight 0.0
  --e2e-v1-asr-kl-weight 0.0
  --e2e-phase3-kl-weight 0.25
  --e2e-commit-weight 0.0
  --e2e-boundary-eos-weight 0.10
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
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS_LOCAL}"
  --lr "${ADAPTER_LR}"
  --min-lr 3e-6
  --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-iters "${TRAIN_ITERS_LOCAL}"
  --lr-decay-style cosine
  --dataloader-type cyclic
  --no-data-sharding
  # Packed E2E records contain many tensor-backed fields. With four arms
  # running concurrently, multiprocessing queues can retain thousands of
  # shared-memory file descriptors per rank and fail with an ancdata error.
  # Main-process loading is mmap-backed and preserves sampler/order exactly.
  --num-workers "${DATA_WORKERS}"
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 0.5
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --dist-ckpt-strictness log_all
  --finetune
  --no-load-optim
  --no-load-rng
  --load "${STAGE_A_LOAD}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --eval-iters "${EVAL_ITERS_LOCAL}"
  --eval-interval "${TRAIN_ITERS_LOCAL}"
  --eval-micro-batch-size 1
  --eval-global-batch-size 2
  --log-interval "${LOG_INTERVAL}"
  --tensorboard-dir "${TB_DIR}"
  --tensorboard-log-interval "${LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${LOG_INTERVAL}"
  --log-world-size-to-tensorboard
  --log-throughput
  --seed "${SEED}"
)

printf '%q ' "${CMD[@]}" > "${LOG}.command"
printf '\n' >> "${LOG}.command"
MONITOR=${LOG%.log}.gpu.csv
(
  echo timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w,power_limit_w
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
    sleep 5
  done
) > "${MONITOR}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
"${CMD[@]}" 2>&1 | tee "${LOG}"
