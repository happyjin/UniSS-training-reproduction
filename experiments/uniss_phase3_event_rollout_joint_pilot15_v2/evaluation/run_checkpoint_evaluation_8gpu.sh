#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"

ITERATION="${ITERATION:?Set ITERATION to a completed checkpoint iteration}"
RUN_NAME="${RUN_NAME:-uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/${RUN_NAME}/iter_$(printf '%07d' "${ITERATION}")}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}"
FORMAL_ROOT="${FORMAL_ROOT:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${FORMAL_ROOT}/formal_train_manifest.jsonl}"
VALID_SOURCE="${VALID_SOURCE:-${FORMAL_ROOT}/formal_valid_manifest.jsonl}"
TRAIN_SAMPLES_PER_DIRECTION="${TRAIN_SAMPLES_PER_DIRECTION:-64}"
VALID_SAMPLES_PER_DIRECTION="${VALID_SAMPLES_PER_DIRECTION:-}"
PARITY_SAMPLES_PER_DIRECTION="${PARITY_SAMPLES_PER_DIRECTION:-2}"
SEED="${SEED:-20260812}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-${USER_ROOT}/conda_envs/uniss-offline-demo/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python}"
EVAL_TAG="${EVAL_TAG:-exact_runtime_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/checkpoint_evaluation/${EVAL_TAG}/iter_$(printf '%07d' "${ITERATION}")}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/runtime_exports/iter_$(printf '%07d' "${ITERATION}")}"

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

for required in "${CHECKPOINT}/.metadata" "${BASE_MODEL}/config.json" \
  "${TRAIN_SOURCE}" "${TRAIN_SOURCE}.offsets.bin" \
  "${VALID_SOURCE}" "${VALID_SOURCE}.offsets.bin" \
  "${INFERENCE_PYTHON}" "${TRAIN_PYTHON}"; do
  [[ -e "${required}" ]] || { echo "Missing evaluation input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/logs"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests"

valid_selection_args=()
if [[ -n "${VALID_SAMPLES_PER_DIRECTION}" ]]; then
  [[ "${VALID_SAMPLES_PER_DIRECTION}" =~ ^[1-9][0-9]*$ ]] || {
    echo "VALID_SAMPLES_PER_DIRECTION must be empty or a positive integer" >&2
    exit 2
  }
  valid_selection_args+=(--samples-per-direction "${VALID_SAMPLES_PER_DIRECTION}")
fi
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.prepare_runtime_manifests \
  --source "${VALID_SOURCE}" --output-root "${MANIFEST_ROOT}/valid_full8" \
  --split valid --num-shards 8 --seed "${SEED}" "${valid_selection_args[@]}" \
  >"${OUTPUT_ROOT}/logs/prepare_valid.log" 2>&1
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.prepare_runtime_manifests \
  --source "${TRAIN_SOURCE}" --output-root "${MANIFEST_ROOT}/train_balanced8" \
  --split train --num-shards 8 --samples-per-direction "${TRAIN_SAMPLES_PER_DIRECTION}" \
  --seed "${SEED}" >"${OUTPUT_ROOT}/logs/prepare_train.log" 2>&1
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.prepare_runtime_manifests \
  --source "${VALID_SOURCE}" --output-root "${MANIFEST_ROOT}/parity_balanced1" \
  --split valid --num-shards 1 --samples-per-direction "${PARITY_SAMPLES_PER_DIRECTION}" \
  --seed "${SEED}" >"${OUTPUT_ROOT}/logs/prepare_parity.log" 2>&1

# Export exactly once before workers start.  Each worker verifies and reuses the
# immutable checksummed export, avoiding distributed-checkpoint export races.
"${INFERENCE_PYTHON}" -m web_demo.true_subsecond_pilot15_streaming_v1.checkpoint_export \
  --checkpoint "${CHECKPOINT}" --base-model "${BASE_MODEL}" --output "${EXPORT_ROOT}" \
  >"${OUTPUT_ROOT}/logs/export.log" 2>&1

wait_for_workers() {
  local phase="$1"
  shift
  local status=0
  local pid
  for pid in "$@"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${phase} worker failed; inspect ${OUTPUT_ROOT}/logs/${phase}_*.log" >&2
    return "${status}"
  fi
}

run_eight_shards() {
  local split="$1"
  local manifest_dir="$2"
  local output_dir="$3"
  local pids=()
  local index gpu manifest output
  for ((index = 0; index < 8; index++)); do
    gpu="${GPU_IDS[${index}]}"
    manifest="${manifest_dir}/part-$(printf '%03d' "${index}").jsonl"
    output="${output_dir}/part-$(printf '%03d' "${index}")"
    CUDA_VISIBLE_DEVICES="${gpu}" ITERATION="${ITERATION}" RUN_NAME="${RUN_NAME}" \
      CHECKPOINT="${CHECKPOINT}" BASE_MODEL="${BASE_MODEL}" EXPORT_ROOT="${EXPORT_ROOT}" \
      FORMAL_MANIFEST="${manifest}" SPEAKER_FORMAL_MANIFEST="${TRAIN_SOURCE}" \
      SPLIT="${split}" SAMPLES=2147483647 DEVICE=cuda:0 FUSE_TICKS=1 STATIC_CACHE=1 \
      TAG="${EVAL_TAG}" OUTPUT="${output}" INFERENCE_PYTHON="${INFERENCE_PYTHON}" \
      "${EVAL_DIR}/evaluate_checkpoint.sh" \
      >"${OUTPUT_ROOT}/logs/${split}_${index}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers "${split}" "${pids[@]}"
}

run_eight_shards valid "${MANIFEST_ROOT}/valid_full8" "${OUTPUT_ROOT}/valid_parts"

valid_summaries=()
valid_manifests=()
for ((index = 0; index < 8; index++)); do
  valid_summaries+=(--summary "${OUTPUT_ROOT}/valid_parts/part-$(printf '%03d' "${index}")/summary.json")
  valid_manifests+=(--expected-manifest "${MANIFEST_ROOT}/valid_full8/part-$(printf '%03d' "${index}").jsonl")
done
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.aggregate_runtime \
  "${valid_summaries[@]}" "${valid_manifests[@]}" \
  --output-root "${OUTPUT_ROOT}/valid_aggregate" \
  >"${OUTPUT_ROOT}/logs/aggregate_valid.log" 2>&1

run_eight_shards train "${MANIFEST_ROOT}/train_balanced8" "${OUTPUT_ROOT}/train_parts"

train_summaries=()
train_manifests=()
for ((index = 0; index < 8; index++)); do
  train_summaries+=(--summary "${OUTPUT_ROOT}/train_parts/part-$(printf '%03d' "${index}")/summary.json")
  train_manifests+=(--expected-manifest "${MANIFEST_ROOT}/train_balanced8/part-$(printf '%03d' "${index}").jsonl")
done
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.aggregate_runtime \
  "${train_summaries[@]}" "${train_manifests[@]}" \
  --output-root "${OUTPUT_ROOT}/train_aggregate" \
  >"${OUTPUT_ROOT}/logs/aggregate_train.log" 2>&1

parity_manifest="${MANIFEST_ROOT}/parity_balanced1/part-000.jsonl"
parity_modes=(fused_cached fused_uncached unfused_cached unfused_uncached)
parity_fuse=(1 1 0 0)
parity_cache=(1 0 1 0)
parity_pids=()
for ((index = 0; index < 4; index++)); do
  mode="${parity_modes[${index}]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${index}]}" ITERATION="${ITERATION}" RUN_NAME="${RUN_NAME}" \
    CHECKPOINT="${CHECKPOINT}" BASE_MODEL="${BASE_MODEL}" EXPORT_ROOT="${EXPORT_ROOT}" \
    FORMAL_MANIFEST="${parity_manifest}" SPEAKER_FORMAL_MANIFEST="${TRAIN_SOURCE}" \
    SPLIT=valid SAMPLES=2147483647 DEVICE=cuda:0 \
    FUSE_TICKS="${parity_fuse[${index}]}" STATIC_CACHE="${parity_cache[${index}]}" \
    TAG="${EVAL_TAG}_${mode}" OUTPUT="${OUTPUT_ROOT}/parity/${mode}" \
    INFERENCE_PYTHON="${INFERENCE_PYTHON}" "${EVAL_DIR}/evaluate_checkpoint.sh" \
    >"${OUTPUT_ROOT}/logs/parity_${mode}.log" 2>&1 &
  parity_pids+=("$!")
done
wait_for_workers parity "${parity_pids[@]}"

parity_args=()
for mode in "${parity_modes[@]}"; do
  parity_args+=(--mode-summary "${mode}=${OUTPUT_ROOT}/parity/${mode}/summary.json")
done
"${TRAIN_PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.compare_runtime_parity \
  "${parity_args[@]}" --output "${OUTPUT_ROOT}/parity/report.json" \
  >"${OUTPUT_ROOT}/logs/compare_parity.log" 2>&1

printf '%s\n' "${OUTPUT_ROOT}"
