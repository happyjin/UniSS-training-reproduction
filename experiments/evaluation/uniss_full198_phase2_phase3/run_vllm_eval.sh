#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 STAGE HF_CHECKPOINT MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

STAGE="$1"
HF_CHECKPOINT="$2"
MANIFEST="$3"
OUTPUT_ROOT="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# vLLM 0.8.5 defaults to the V1 engine, which rejects per-request logits
# processors. UniSS must mask Megatron's 73 padded vocabulary rows, so use the
# compatible V0 engine until V1 supports this sampling hook.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

if [[ -e "${OUTPUT_ROOT}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/metrics"

VERIFY_ARGS=()
if [[ "${ALLOW_GENERATED_FAILURES:-0}" == "1" ]]; then
  VERIFY_ARGS+=(--allow-generated-failures)
fi

# A 0.5B model does not benefit from four-way tensor parallelism. Split the
# manifest instead so each reserved GPU runs an independent vLLM/decode worker.
GPU_LIST_VALUE="${EVAL_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_EVAL_GPUS="${#GPU_IDS[@]}"
PRIMARY_GPU="${GPU_IDS[0]}"
EXPECTED_RESULTS=$(( $(wc -l < "${MANIFEST}") * 2 ))
SHARD_ROOT="${OUTPUT_ROOT}/data_parallel_shards"

jsonl_count() {
  if [[ -f "$1" ]]; then
    wc -l <"$1"
  else
    echo 0
  fi
}

run_generate_decode_one() {
  local gpu="$1"
  local manifest="$2"
  local output_root="$3"
  local expected=$(( $(wc -l < "${manifest}") * 2 ))
  local generation_file="${output_root}/vllm/generation_results.jsonl"
  local results_file="${output_root}/results.jsonl"
  local generation_count
  local results_count
  generation_count="$(jsonl_count "${generation_file}")"
  results_count="$(jsonl_count "${results_file}")"
  if [[ "${generation_count}" -gt "${expected}" || "${results_count}" -gt "${expected}" ]]; then
    echo "Shard output exceeds expected count: root=${output_root} expected=${expected} generation=${generation_count} results=${results_count}" >&2
    return 1
  fi

  if [[ "${generation_count}" -lt "${expected}" ]]; then
    local generation_resume=()
    if [[ -e "${output_root}/vllm" ]]; then
      generation_resume+=(--resume)
    fi
    # UniSS requires its padded-vocabulary logits processor. vLLM V0 rejects
    # custom logits processors with multi-step decoding, so keep one scheduler
    # step while increasing queue depth and active sequence concurrency.
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.vllm_generate \
      --manifest "${manifest}" \
      --model "${HF_CHECKPOINT}" \
      --output-dir "${output_root}/vllm" \
      --mode quality performance \
      --limit-records 0 \
      --temperature 0.7 \
      --top-p 0.8 \
      --top-k -1 \
      --repetition-penalty 1.1 \
      --max-new-tokens "${MAX_NEW_TOKENS:-1500}" \
      --seed "${SEED:-20260726}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
      --max-model-len "${MAX_MODEL_LEN:-2048}" \
      --max-num-seqs "${MAX_NUM_SEQS:-512}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-65536}" \
      --num-scheduler-steps "${NUM_SCHEDULER_STEPS:-1}" \
      --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE:-2048}" \
      --request-batch-size "${REQUEST_BATCH_SIZE:-2048}" \
      --dtype bfloat16 \
      "${generation_resume[@]}"
  fi

  results_count="$(jsonl_count "${results_file}")"
  if [[ "${results_count}" -lt "${expected}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.decode_audio \
      --input "${generation_file}" \
      --manifest "${manifest}" \
      --speech-tokenizer "${SPEECH_TOKENIZER}" \
      --output-dir "${output_root}" \
      --device cuda:0 \
      --save-source-audio \
      --save-reference-audio \
      --batch-size "${DECODE_BATCH_SIZE:-32}" \
      --resume
  fi
}

CANONICAL_GENERATION_COUNT="$(jsonl_count "${OUTPUT_ROOT}/vllm/generation_results.jsonl")"
CANONICAL_RESULTS_COUNT="$(jsonl_count "${OUTPUT_ROOT}/results.jsonl")"
if [[ "${CANONICAL_GENERATION_COUNT}" -eq "${EXPECTED_RESULTS}" && "${CANONICAL_RESULTS_COUNT}" -eq "${EXPECTED_RESULTS}" ]]; then
  echo "Canonical generation/audio already complete: ${EXPECTED_RESULTS} results"
elif [[ "${NUM_EVAL_GPUS}" -gt 1 && ( -d "${SHARD_ROOT}" || "${CANONICAL_GENERATION_COUNT}" -eq 0 ) ]]; then
  mkdir -p "${SHARD_ROOT}/manifests" "${SHARD_ROOT}/logs"
  "${ENV_ROOT}/bin/python" -m evaluation.shard_manifest \
    --input "${MANIFEST}" \
    --output-dir "${SHARD_ROOT}/manifests" \
    --num-shards "${NUM_EVAL_GPUS}"
  pids=()
  for ((index = 0; index < NUM_EVAL_GPUS; index++)); do
    shard_manifest="${SHARD_ROOT}/manifests/manifest.part_$(printf '%03d' "${index}")-of-$(printf '%03d' "${NUM_EVAL_GPUS}").jsonl"
    shard_output="${SHARD_ROOT}/shard_$(printf '%03d' "${index}")"
    run_generate_decode_one "${GPU_IDS[${index}]}" "${shard_manifest}" "${shard_output}" \
      >"${SHARD_ROOT}/logs/shard_${index}.log" 2>&1 &
    pids+=("$!")
  done
  worker_status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || worker_status=$?
  done
  if [[ "${worker_status}" -ne 0 ]]; then
    echo "A data-parallel generation worker failed:" >&2
    tail -n 100 "${SHARD_ROOT}/logs/shard_"*.log >&2 || true
    exit "${worker_status}"
  fi
  "${ENV_ROOT}/bin/python" -m evaluation.merge_evaluation_shards \
    --manifest "${MANIFEST}" \
    --output-root "${OUTPUT_ROOT}" \
    --shard-root "${SHARD_ROOT}" \
    --num-shards "${NUM_EVAL_GPUS}" \
    --modes quality performance
else
  run_generate_decode_one "${PRIMARY_GPU}" "${MANIFEST}" "${OUTPUT_ROOT}"
fi

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_audio_output.py" \
  --manifest "${MANIFEST}" \
  --results "${OUTPUT_ROOT}/results.jsonl" \
  --summary "${OUTPUT_ROOT}/summary.json" \
  --expected-modes quality performance \
  "${VERIFY_ARGS[@]}"

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/vllm/generation_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/text_bleu.json"

"${ENV_ROOT}/bin/python" -m evaluation.slc_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics"

echo "${STAGE} vLLM output: ${OUTPUT_ROOT}"
