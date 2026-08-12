#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
MODEL_ROOT="${MODEL_ROOT:-/opt/dlami/nvme/jasonleeeli/evaluation_models}"
AUTOPCP_COMPARATOR="${AUTOPCP_COMPARATOR:-${MODEL_ROOT}/AutoPCP-multilingual-v2}"
AUTOPCP_ENCODER="${AUTOPCP_ENCODER:-${MODEL_ROOT}/wav2vec2-large-xlsr-53}"
DEVICE="${DEVICE:-cuda:0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for required in \
  "${AUTOPCP_COMPARATOR}/model.config" \
  "${AUTOPCP_COMPARATOR}/model.pt" \
  "${AUTOPCP_ENCODER}/config.json" \
  "${AUTOPCP_ENCODER}/preprocessor_config.json" \
  "${AUTOPCP_ENCODER}/pytorch_model.bin"; do
  [[ -f "${required}" ]] || { echo "Missing objective-metric model file: ${required}" >&2; exit 1; }
done

export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/opt/dlami/nvme/jasonleeeli/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/opt/dlami/nvme/jasonleeeli/cache/torch}"

mkdir -p "${OUTPUT_ROOT}/metrics"

GPU_LIST_VALUE="${EVAL_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_SHARDS="${METRIC_NUM_GPUS:-${#GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -lt 1 || "${NUM_SHARDS}" -gt "${#GPU_IDS[@]}" ]]; then
  echo "METRIC_NUM_GPUS=${NUM_SHARDS} is incompatible with EVAL_GPU_LIST=${GPU_LIST_VALUE}" >&2
  exit 2
fi
SHARD_ROOT="${OUTPUT_ROOT}/metrics/shards"
mkdir -p "${SHARD_ROOT}/logs"

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
    tail -n 80 "${SHARD_ROOT}/logs/${metric}_"*.log >&2 || true
    return "${status}"
  fi
}

run_asr_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/asr_results.jsonl"
  local completed_args=()
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  local pids=()
  local index gpu output
  mkdir -p "${SHARD_ROOT}/asr"
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    output="${SHARD_ROOT}/asr/part_$(printf '%03d' "${index}").jsonl"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.asr_transcribe \
      --input "${OUTPUT_ROOT}/results.jsonl" \
      --output "${output}" \
      --device cuda:0 \
      --batch-size "${ASR_BATCH_SIZE:-8}" \
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
    --input "${OUTPUT_ROOT}/results.jsonl" \
    --metric-dir "${OUTPUT_ROOT}/metrics" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_SHARDS}"
}

run_utmos_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/per_sample_utmos.jsonl"
  local completed_args=()
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  local pids=()
  local index gpu part_dir
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    part_dir="${SHARD_ROOT}/utmos/part_$(printf '%03d' "${index}")"
    mkdir -p "${part_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.utmos_metrics \
      --input "${OUTPUT_ROOT}/results.jsonl" \
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
    --input "${OUTPUT_ROOT}/results.jsonl" \
    --metric-dir "${OUTPUT_ROOT}/metrics" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_SHARDS}"
}

run_autopcp_shards() {
  local canonical="${OUTPUT_ROOT}/metrics/per_sample_autopcp.jsonl"
  local completed_args=()
  if [[ -f "${canonical}" ]]; then
    completed_args+=(--completed-input "${canonical}")
  fi
  local pids=()
  local index gpu part_dir
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_IDS[${index}]}"
    part_dir="${SHARD_ROOT}/autopcp/part_$(printf '%03d' "${index}")"
    mkdir -p "${part_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.autopcp_metrics \
      --input "${OUTPUT_ROOT}/results.jsonl" \
      --output-dir "${part_dir}" \
      --comparator-path "${AUTOPCP_COMPARATOR}" \
      --encoder-model "${AUTOPCP_ENCODER}" \
      --device cuda:0 \
      --pick-layer 9 \
      --symmetrize \
      --batch-size "${AUTOPCP_BATCH_SIZE:-16}" \
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
    --input "${OUTPUT_ROOT}/results.jsonl" \
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
