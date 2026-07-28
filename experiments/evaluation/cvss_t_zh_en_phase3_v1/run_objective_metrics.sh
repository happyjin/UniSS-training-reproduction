#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 OUTPUT_ROOT EXPECTED_DIRECTION" >&2
  echo "EXPECTED_DIRECTION must be cmn->eng or eng->cmn" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
EXPECTED_DIRECTION="$2"
case "${EXPECTED_DIRECTION}" in
  cmn-\>eng | eng-\>cmn) ;;
  *) echo "Unsupported CVSS-T direction: ${EXPECTED_DIRECTION}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
MODEL_ROOT="${MODEL_ROOT:-/opt/dlami/nvme/jasonleeeli/evaluation_models}"
AUTOPCP_COMPARATOR="${AUTOPCP_COMPARATOR:-${MODEL_ROOT}/AutoPCP-multilingual-v2}"
EXPECTED_PAIRS="${EXPECTED_PAIRS:-4897}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/opt/dlami/nvme/jasonleeeli/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/opt/dlami/nvme/jasonleeeli/cache/torch}"

RESULTS_PATH="${OUTPUT_ROOT}/results.jsonl"
[[ -f "${RESULTS_PATH}" ]] || { echo "Missing decoded CVSS-T results: ${RESULTS_PATH}" >&2; exit 1; }
[[ -d "${AUTOPCP_COMPARATOR}" ]] || { echo "Missing AutoPCP comparator: ${AUTOPCP_COMPARATOR}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}/metrics"

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_SHARDS="${METRIC_NUM_GPUS:-${#GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -lt 1 || "${NUM_SHARDS}" -gt "${#GPU_IDS[@]}" ]]; then
  echo "METRIC_NUM_GPUS=${NUM_SHARDS} is incompatible with EVAL_GPU_LIST=${GPU_LIST_VALUE}" >&2
  exit 2
fi
SHARD_ROOT="${OUTPUT_ROOT}/metrics/shards"
mkdir -p "${SHARD_ROOT}/logs"

integrity_args=()
if [[ "${ALLOW_GENERATED_FAILURES:-0}" == "1" ]]; then
  integrity_args+=(--allow-generated-failures)
fi
"${ENV_ROOT}/bin/python" -m evaluation.cvss_t.validate_results \
  --input "${RESULTS_PATH}" \
  --output "${OUTPUT_ROOT}/metrics/result_integrity.json" \
  --expected-pairs "${EXPECTED_PAIRS}" \
  --expected-direction "${EXPECTED_DIRECTION}" \
  --modes quality performance \
  "${integrity_args[@]}"

# CPU-only metrics are evaluated before GPU model loading. They use the
# generated translation and official canonical source waveform respectively.
"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/vllm/generation_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/text_bleu.json" \
  --hypothesis-field generated_translation \
  --reference-field translation_ref
"${ENV_ROOT}/bin/python" -m evaluation.slc_metrics \
  --input "${RESULTS_PATH}" \
  --output-dir "${OUTPUT_ROOT}/metrics"

wait_for_workers() {
  local metric="$1"
  shift
  local status=0
  local pid
  for pid in "$@"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${metric} metric shard failed; recent worker logs:" >&2
    tail -n 100 "${SHARD_ROOT}/logs/${metric}_"*.log >&2 || true
    return "${status}"
  fi
}

run_asr_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/asr_results.jsonl"
  local completed_args=()
  local pids=()
  local index gpu output
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  mkdir -p "${SHARD_ROOT}/asr"
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    output="${SHARD_ROOT}/asr/part_$(printf '%03d' "${index}").jsonl"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.asr_transcribe \
      --input "${RESULTS_PATH}" \
      --output "${output}" \
      --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
      --paraformer-model "${PARAFORMER_MODEL:-iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch}" \
      --device cuda:0 \
      --batch-size "${ASR_BATCH_SIZE:-16}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${index}" \
      --resume \
      "${completed_args[@]}" \
      >"${SHARD_ROOT}/logs/asr_${index}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers asr "${pids[@]}"
  "${ENV_ROOT}/bin/python" -m evaluation.merge_metric_shards \
    --metric asr \
    --input "${RESULTS_PATH}" \
    --metric-dir "${OUTPUT_ROOT}/metrics" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_SHARDS}"
}

run_utmos_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/per_sample_utmos.jsonl"
  local completed_args=()
  local pids=()
  local index gpu part_dir
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    part_dir="${SHARD_ROOT}/utmos/part_$(printf '%03d' "${index}")"
    mkdir -p "${part_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.utmos_metrics \
      --input "${RESULTS_PATH}" \
      --output-dir "${part_dir}" \
      --device cuda:0 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${index}" \
      --resume \
      "${completed_args[@]}" \
      >"${SHARD_ROOT}/logs/utmos_${index}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers utmos "${pids[@]}"
  "${ENV_ROOT}/bin/python" -m evaluation.merge_metric_shards \
    --metric utmos \
    --input "${RESULTS_PATH}" \
    --metric-dir "${OUTPUT_ROOT}/metrics" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_SHARDS}"
}

run_autopcp_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/per_sample_autopcp.jsonl"
  local completed_args=()
  local pids=()
  local index gpu part_dir
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    part_dir="${SHARD_ROOT}/autopcp/part_$(printf '%03d' "${index}")"
    mkdir -p "${part_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.autopcp_metrics \
      --input "${RESULTS_PATH}" \
      --output-dir "${part_dir}" \
      --comparator-path "${AUTOPCP_COMPARATOR}" \
      --device cuda:0 \
      --pick-layer 9 \
      --symmetrize \
      --batch-size "${AUTOPCP_BATCH_SIZE:-32}" \
      --chunk-size "${AUTOPCP_CHUNK_SIZE:-1024}" \
      --num-process "${AUTOPCP_NUM_PROCESS:-1}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${index}" \
      --resume \
      "${completed_args[@]}" \
      >"${SHARD_ROOT}/logs/autopcp_${index}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers autopcp "${pids[@]}"
  "${ENV_ROOT}/bin/python" -m evaluation.merge_metric_shards \
    --metric autopcp \
    --input "${RESULTS_PATH}" \
    --metric-dir "${OUTPUT_ROOT}/metrics" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_SHARDS}"
}

run_asr_shards
"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/metrics/asr_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/speech_bleu.json" \
  --hypothesis-field asr_text \
  --reference-field translation_ref
run_utmos_shards
run_autopcp_shards

echo "CVSS-T Table 1 objective metrics completed: ${OUTPUT_ROOT}"
