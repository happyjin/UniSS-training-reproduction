#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_V2_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_v2_causal_15shard_v1.env}"
MODE="${1:-clone}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1
[[ "${MODE}" == "clone" || "${MODE}" == "prefix80" || "${MODE}" == "smoke" ]] || {
  echo "mode must be clone, prefix80, or smoke" >&2
  exit 2
}
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${STAGE_B_V2_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${STAGE_B_V2_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${STAGE_B_V2_CPU_THREADS}}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

if [[ "${MODE}" == "clone" ]]; then
  sidecar_train="${CLONE_TRAIN_ROOT}/manifest.jsonl"
  sidecar_valid="${CLONE_VALID_ROOT}/manifest.jsonl"
  output_dir="${STAGE_B_V2_CLONE_ROOT}"
  tensorboard_dir="${STAGE_B_V2_CLONE_RUN_ROOT}/tensorboard"
  log_root="${STAGE_B_V2_CLONE_LOG_ROOT}"
  steps="${STAGE_B_V2_CLONE_STEPS}"
  learning_rate="${STAGE_B_V2_CLONE_LR}"
  initialize_args=()
  nproc=8
elif [[ "${MODE}" == "prefix80" ]]; then
  sidecar_train="${PREFIX_TRAIN_ROOT}/manifest.jsonl"
  sidecar_valid="${PREFIX_VALID_ROOT}/manifest.jsonl"
  output_dir="${STAGE_B_V2_PREFIX_ROOT}"
  tensorboard_dir="${STAGE_B_V2_PREFIX_RUN_ROOT}/tensorboard"
  log_root="${STAGE_B_V2_PREFIX_LOG_ROOT}"
  steps="${STAGE_B_V2_PREFIX_STEPS}"
  learning_rate="${STAGE_B_V2_PREFIX_LR}"
  initialize_args=(--initialize-from "${STAGE_B_V2_CLONE_ROOT}/best.pt")
  nproc=8
else
  sidecar_train="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/smoke/stage_a_v3_clone_16_v2/manifest.jsonl"
  sidecar_valid="${sidecar_train}"
  output_dir="${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v2/smoke/stage_b_v2_launcher"
  tensorboard_dir="${REPO_ROOT}/runs/simul_uniss_subsecond_v2/smoke/stage_b_v2_launcher/tensorboard"
  log_root="${REPO_ROOT}/logs/simul_uniss_subsecond_v2/smoke/stage_b_v2_launcher"
  steps=2
  learning_rate=1e-4
  initialize_args=()
  nproc=1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  for path in \
    "${sidecar_train}" "${sidecar_train}.offsets.bin" \
    "${sidecar_valid}" "${sidecar_valid}.offsets.bin" \
    "${SOURCE_TRAIN_MANIFEST}" "${SOURCE_VALID_MANIFEST}" \
    "${POLICY_TOKENIZER_MODEL}" "${WHISPERVQ_MODEL}/model.safetensors"; do
    [[ -f "${path}" ]] || { echo "Missing Stage-B-v2 input: ${path}" >&2; exit 1; }
  done
  if [[ "${MODE}" == "prefix80" ]]; then
    [[ -f "${STAGE_B_V2_CLONE_ROOT}/best.pt" ]] || {
      echo "Missing completed clone pretrain checkpoint" >&2
      exit 1
    }
  fi
fi

model_args=(
  --hidden-size "${STAGE_B_V2_HIDDEN_SIZE}"
  --num-layers "${STAGE_B_V2_NUM_LAYERS}"
  --num-heads "${STAGE_B_V2_NUM_HEADS}"
  --ffn-dim "${STAGE_B_V2_FFN_DIM}"
)
if [[ "${MODE}" == "smoke" ]]; then
  model_args=(--hidden-size 128 --num-layers 2 --num-heads 4 --ffn-dim 512)
fi

command=(
  torchrun --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}"
  --master-addr 127.0.0.1 --master-port "${STAGE_B_V2_MASTER_PORT}"
  -m training.simul_uniss.subsecond_v2.train_stage_b_v2
  --sidecar-manifest "${sidecar_train}"
  --valid-sidecar-manifest "${sidecar_valid}"
  --source-manifest "${SOURCE_TRAIN_MANIFEST}"
  --valid-source-manifest "${SOURCE_VALID_MANIFEST}"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --codebook-model "${WHISPERVQ_MODEL}"
  --output-dir "${output_dir}"
  --tensorboard-dir "${tensorboard_dir}"
  --device cuda --bf16
  --batch-size "$([[ "${MODE}" == "smoke" ]] && echo 2 || echo "${STAGE_B_V2_BATCH_SIZE}")"
  --max-steps "${steps}"
  --learning-rate "${learning_rate}"
  --num-workers "$([[ "${MODE}" == "smoke" ]] && echo 0 || echo "${STAGE_B_V2_NUM_WORKERS}")"
  --eval-interval "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo 500)"
  --eval-batches "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo 8)"
  --save-interval "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo 500)"
  --log-interval "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo 10)"
  --representation-only-steps "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo "${STAGE_B_V2_REPRESENTATION_ONLY_STEPS}")"
  --auxiliary-ramp-steps "$([[ "${MODE}" == "smoke" ]] && echo 1 || echo "${STAGE_B_V2_AUXILIARY_RAMP_STEPS}")"
  --quantize-chunk-size "${STAGE_B_V2_QUANTIZE_CHUNK_SIZE}"
  "${model_args[@]}" "${initialize_args[@]}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICES}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${output_dir}" "${tensorboard_dir}" "${log_root}"
monitor_log="${log_root}/gpu_monitor.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader,nounits --loop=2 > "${monitor_log}" &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${command[@]}" 2>&1 | tee "${log_root}/train.log"
