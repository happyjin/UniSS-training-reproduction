#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 RUN_ID ADAPTER_CHECKPOINT OUTPUT_DIR [WORKERS]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
RUN_ID=$1
ADAPTER_CHECKPOINT=$2
OUTPUT_DIR=$3
WORKERS=${4:-8}
if ! [[ "${WORKERS}" =~ ^[1-8]$ ]]; then
  echo "WORKERS must be 1..8" >&2
  exit 2
fi
[[ -f "${ADAPTER_CHECKPOINT}/.metadata" ]] || {
  echo "missing adapter checkpoint: ${ADAPTER_CHECKPOINT}" >&2
  exit 3
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "refusing to overwrite routed evaluation: ${OUTPUT_DIR}" >&2
  exit 3
}

SELECTION=${REPO_ROOT}/reports/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/free_running_gates/free_running_gate_learning100u_20260821T142900Z/SELECTION.json
GOLD=${REPO_ROOT}/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl
BASE_HF=${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf
PHASE3_HF=${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
V1_CHECKPOINT=${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381
BICODEC=${REPO_ROOT}/pretrained_models/UniSS/bicodec/BiCodec

mkdir -p "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/logs"
export HF_HOME=${USER_ROOT}/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON_BIN}"):${PATH}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_KERNEL_CACHE_PATH=${USER_ROOT}/.cache/torch_kernels
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}" "${TMPDIR}"

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  report=$(printf '%s/workers/worker_%02d.json' "${OUTPUT_DIR}" "${worker}")
  audio=$(printf '%s/audio/worker_%02d' "${OUTPUT_DIR}" "${worker}")
  log=$(printf '%s/logs/worker_%02d.log' "${OUTPUT_DIR}" "${worker}")
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON_BIN}" -u \
    "${EXPERIMENT_ROOT}/evaluation/run_worker.py" \
    --run-id "${RUN_ID}" \
    --selection "${SELECTION}" \
    --gold "${GOLD}" \
    --base-hf "${BASE_HF}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --phase3-hf "${PHASE3_HF}" \
    --v1-checkpoint "${V1_CHECKPOINT}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC}" \
    --worker-index "${worker}" \
    --num-workers "${WORKERS}" \
    --report "${report}" \
    --audio-dir "${audio}" \
    --device cuda:0 > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more routed workers failed" >&2
  exit 1
fi

aggregate=(
  "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/aggregate.py"
  --selection "${SELECTION}"
  --output "${OUTPUT_DIR}/SUMMARY.json"
)
for ((worker=0; worker<WORKERS; worker++)); do
  aggregate+=(--worker-report "$(printf '%s/workers/worker_%02d.json' "${OUTPUT_DIR}" "${worker}")")
done
"${aggregate[@]}" > "${OUTPUT_DIR}/aggregate.stdout.json"

