#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"

ITERATION="${ITERATION:?Set ITERATION to the selected streaming checkpoint iteration}"
RUN_NAME="${RUN_NAME:-uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/runtime_exports/iter_$(printf '%07d' "${ITERATION}")}"
FORMAL_ROOT="${FORMAL_ROOT:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal}"
VALID_SOURCE="${VALID_SOURCE:-${FORMAL_ROOT}/formal_valid_manifest.jsonl}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-${USER_ROOT}/conda_envs/uniss-offline-demo/bin/python}"
SAMPLES_PER_DIRECTION="${SAMPLES_PER_DIRECTION:-64}"
SEED="${SEED:-20260812}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TAG="${TAG:-phase3_retention_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/phase3_retention/${TAG}/iter_$(printf '%07d' "${ITERATION}")}"

for required in "${EXPORT_ROOT}/manifest.json" "${VALID_SOURCE}" "${VALID_SOURCE}.offsets.bin" \
  "${SPEECH_TOKENIZER}" "${TRAIN_PYTHON}" "${INFERENCE_PYTHON}"; do
  [[ -e "${required}" ]] || { echo "Missing retention input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }
[[ "${SAMPLES_PER_DIRECTION}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SAMPLES_PER_DIRECTION must be positive" >&2
  exit 2
}

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
  echo "GPU_LIST must contain exactly 8 GPU IDs" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu}" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || { echo "Duplicate GPU ID: ${gpu}" >&2; exit 2; }
  SEEN_GPUS["${gpu}"]=1
done

mkdir -p "${OUTPUT_ROOT}/logs"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests/valid_balanced8"
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.prepare_runtime_manifests \
  --source "${VALID_SOURCE}" --output-root "${MANIFEST_ROOT}" \
  --split valid --num-shards 8 --samples-per-direction "${SAMPLES_PER_DIRECTION}" \
  --seed "${SEED}" >"${OUTPUT_ROOT}/logs/prepare.log" 2>&1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/cache/pytorch/kernels}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${USER_ROOT}/cache/cuda}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}" "${CUDA_CACHE_PATH}" "${TMPDIR}"

pids=()
for ((index = 0; index < 8; index++)); do
  gpu="${GPU_IDS[${index}]}"
  manifest="${MANIFEST_ROOT}/part-$(printf '%03d' "${index}").jsonl"
  part="${OUTPUT_ROOT}/parts/part-$(printf '%03d' "${index}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${INFERENCE_PYTHON}" \
    -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.evaluate_phase3_retention \
    --formal-manifest "${manifest}" --export "${EXPORT_ROOT}" \
    --speech-tokenizer "${SPEECH_TOKENIZER}" --output "${part}" --device cuda:0 \
    >"${OUTPUT_ROOT}/logs/retention_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  echo "Phase3 retention worker failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

parts=()
for ((index = 0; index < 8; index++)); do
  parts+=(--part "${OUTPUT_ROOT}/parts/part-$(printf '%03d' "${index}")/results.jsonl")
done
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.merge_phase3_retention \
  "${parts[@]}" --output-root "${OUTPUT_ROOT}/aggregate" \
  >"${OUTPUT_ROOT}/logs/merge.log" 2>&1

printf '%s\n' "${OUTPUT_ROOT}"
