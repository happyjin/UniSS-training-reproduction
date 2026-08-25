#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 RUN_ID ADAPTER_CHECKPOINT_OR_NONE OUTPUT_DIR GPU0 GPU1" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
RUN_ID=$1
ADAPTER=$2
OUTPUT=$3
GPU0=$4
GPU1=$5
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

SELECTION=${EXPERIMENT_ROOT}/evaluation/protocols/validation64_e2e16_seed20260825.json
VALID_MANIFEST=${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_valid_manifest.jsonl
LEGACY_RESULTS=${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/chunk_640ms_v1/results.json
BASE_HF=${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf
V1_CHECKPOINT=${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381
SNAPSHOT=${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json
STRICT_RUNTIME=${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/run_strict_causal_cascade.py

run_chunk() {
  local chunk=$1 gpu=$2
  local adapter_args=()
  [[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" -u \
    "${EXPERIMENT_ROOT}/evaluation/strict_cascade.py" \
    --run-id "${RUN_ID}" \
    --decision-chunk-ms "${chunk}" \
    --output "${OUTPUT}/chunk_${chunk}ms" \
    --base-hf "${BASE_HF}" \
    "${adapter_args[@]}" \
    --v1-checkpoint "${V1_CHECKPOINT}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
    --source-snapshot "${SNAPSHOT}" \
    --strict-runtime "${STRICT_RUNTIME}" \
    --selection "${SELECTION}" \
    --validation-manifest "${VALID_MANIFEST}" \
    --validation-sample-id emilia_zh_0007320789 \
    --validation-sample-id emilia_zh_0004102476 \
    --validation-sample-id emilia_zh_0006809993 \
    --validation-sample-id emilia_zh_0004751176 \
    --validation-sample-id emilia_zh_0004647293 \
    --validation-sample-id EN_B00083_S01688_W000003 \
    --legacy-results "${LEGACY_RESULTS}" \
    --legacy-sample-id NCSSD_R_EN_0000000000 \
    --legacy-sample-id HQ-Conversations_0000000009 \
    --device cuda:0 > "${OUTPUT}/logs/chunk_${chunk}ms.log" 2>&1
}

(run_chunk 160 "${GPU0}"; run_chunk 640 "${GPU0}") & left=$!
(run_chunk 320 "${GPU1}"; run_chunk 1280 "${GPU1}") & right=$!
status=0
wait "${left}" || status=1
wait "${right}" || status=1
[[ "${status}" -eq 0 ]] || { echo "short audio suite failed" >&2; exit 1; }
"${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/aggregate_listening.py" \
  --root "${OUTPUT}" --output "${OUTPUT}/SUMMARY.json" \
  > "${OUTPUT}/aggregate.stdout.json"

