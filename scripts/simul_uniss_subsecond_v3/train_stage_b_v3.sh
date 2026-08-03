#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${V3_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${V3_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${V3_CPU_THREADS}}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

for path in \
  "${V3_MIXED_TRAIN_MANIFEST}" "${V3_MIXED_TRAIN_MANIFEST}.offsets.bin" \
  "${V3_MIXED_VALID_MANIFEST}" "${V3_MIXED_VALID_MANIFEST}.offsets.bin" \
  "${SOURCE_TRAIN_MANIFEST}" "${SOURCE_VALID_MANIFEST}" \
  "${POLICY_TOKENIZER_MODEL}" "${WHISPERVQ_MODEL}/model.safetensors" \
  "${V2_PREFIX_CHECKPOINT}"; do
  if [[ "${DRY_RUN}" -eq 0 && ! -f "${path}" ]]; then
    echo "Missing Stage-B-v3 input: ${path}" >&2
    exit 1
  fi
done

command=(
  torchrun --nnodes 1 --node-rank 0 --nproc-per-node 8
  --master-addr 127.0.0.1 --master-port "${V3_MASTER_PORT}"
  -m training.simul_uniss.subsecond_v3.train_stage_b_v3
  --sidecar-manifest "${V3_MIXED_TRAIN_MANIFEST}"
  --valid-sidecar-manifest "${V3_MIXED_VALID_MANIFEST}"
  --source-manifest "${SOURCE_TRAIN_MANIFEST}"
  --valid-source-manifest "${SOURCE_VALID_MANIFEST}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --codebook-model "${WHISPERVQ_MODEL}"
  --initialize-from "${V2_PREFIX_CHECKPOINT}"
  --output-dir "${V3_CHECKPOINT_ROOT}"
  --tensorboard-dir "${V3_RUN_ROOT}/tensorboard"
  --device cuda --bf16
  --batch-size "${V3_BATCH_SIZE}"
  --max-steps "${V3_STEPS}"
  --learning-rate "${V3_LEARNING_RATE}"
  --num-workers "${V3_NUM_WORKERS}"
  --eval-interval 500 --eval-batches 8 --log-interval 10
  --representation-only-steps 2000 --auxiliary-ramp-steps 2000
  --quantize-chunk-size 256 --keep-top-k 3
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICES}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${V3_CHECKPOINT_ROOT}" "${V3_RUN_ROOT}/tensorboard" "${V3_LOG_ROOT}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader,nounits --loop=2 > "${V3_LOG_ROOT}/gpu_monitor.csv" &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${command[@]}" 2>&1 | tee "${V3_LOG_ROOT}/train.log"
