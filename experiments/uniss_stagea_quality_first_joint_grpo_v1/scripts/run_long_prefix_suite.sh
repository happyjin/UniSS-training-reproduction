#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "Usage: $0 RUN_ID ADAPTER_CHECKPOINT_OR_NONE OUTPUT_DIR GPU0 GPU1 [GPU2 GPU3]" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
RUN_ID=$1
ADAPTER=$2
OUTPUT=$3
GPU0=$4
GPU1=$5
GPU2=${6:-${GPU0}}
GPU3=${7:-${GPU1}}
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
if [[ "${ADAPTER}" != NONE ]]; then
  [[ -f "${ADAPTER}/.metadata" ]] || { echo "missing adapter checkpoint" >&2; exit 3; }
fi
mkdir -p "${OUTPUT}/logs"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

run_chunk() {
  local chunk=$1 gpu=$2
  local adapter_args=()
  [[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" -u \
    "${EXPERIMENT_ROOT}/evaluation/strict_cascade.py" \
    --run-id "${RUN_ID}" \
    --decision-chunk-ms "${chunk}" \
    --output "${OUTPUT}/chunk_${chunk}ms" \
    --base-hf "${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf" \
    "${adapter_args[@]}" \
    --v1-checkpoint "${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
    --source-snapshot "${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json" \
    --strict-runtime "${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/run_strict_causal_cascade.py" \
    --external-audio-protocol "${EXPERIMENT_ROOT}/evaluation/protocols/long_audio4_prefix60.json" \
    --device cuda:0 > "${OUTPUT}/logs/chunk_${chunk}ms.log" 2>&1
}

(run_chunk 160 "${GPU0}") & p0=$!
(run_chunk 320 "${GPU1}") & p1=$!
(run_chunk 640 "${GPU2}") & p2=$!
(run_chunk 1280 "${GPU3}") & p3=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
wait "${p3}" || status=1
[[ "${status}" -eq 0 ]] || { echo "long prefix suite failed" >&2; exit 1; }
"${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/aggregate_listening.py" \
  --root "${OUTPUT}" --output "${OUTPUT}/SUMMARY.json" \
  > "${OUTPUT}/aggregate.stdout.json"
