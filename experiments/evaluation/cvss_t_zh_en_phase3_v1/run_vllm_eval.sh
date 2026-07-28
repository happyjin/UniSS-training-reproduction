#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

MANIFEST="$1"
OUTPUT_ROOT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

[[ -f "${MANIFEST}" ]] || { echo "Missing CVSS manifest: ${MANIFEST}" >&2; exit 1; }
[[ -d "${HF_CHECKPOINT}" ]] || { echo "Missing Phase3 HF checkpoint: ${HF_CHECKPOINT}" >&2; exit 1; }
if [[ -e "${OUTPUT_ROOT}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite CVSS evaluation output: ${OUTPUT_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_GPUS="${#GPU_IDS[@]}"
SHARD_ROOT="${OUTPUT_ROOT}/data_parallel_shards"
mkdir -p "${SHARD_ROOT}/manifests" "${SHARD_ROOT}/logs"
"${ENV_ROOT}/bin/python" -m evaluation.shard_manifest \
  --input "${MANIFEST}" \
  --output-dir "${SHARD_ROOT}/manifests" \
  --num-shards "${NUM_GPUS}"

run_one() {
  local index="$1"
  local gpu="$2"
  local shard_manifest
  local shard_output
  local resume_args=()
  local decode_resume=()
  shard_manifest="${SHARD_ROOT}/manifests/manifest.part_$(printf '%03d' "${index}")-of-$(printf '%03d' "${NUM_GPUS}").jsonl"
  shard_output="${SHARD_ROOT}/shard_$(printf '%03d' "${index}")"
  if [[ -d "${shard_output}/vllm" ]]; then
    resume_args+=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.vllm_generate \
    --manifest "${shard_manifest}" \
    --model "${HF_CHECKPOINT}" \
    --output-dir "${shard_output}/vllm" \
    --mode quality performance \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k -1 \
    --repetition-penalty 1.1 \
    --max-new-tokens "${MAX_NEW_TOKENS:-1500}" \
    --seed "${SEED:-20260728}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-model-len "${MAX_MODEL_LEN:-2048}" \
    --max-num-seqs "${MAX_NUM_SEQS:-512}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-65536}" \
    --num-scheduler-steps 1 \
    --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE:-2048}" \
    --request-batch-size "${REQUEST_BATCH_SIZE:-2048}" \
    --dtype bfloat16 \
    "${resume_args[@]}"

  if [[ -f "${shard_output}/results.jsonl" ]]; then
    decode_resume+=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.cvss_t.decode_audio \
    --input "${shard_output}/vllm/generation_results.jsonl" \
    --manifest "${shard_manifest}" \
    --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${shard_output}" \
    --device cuda:0 \
    --batch-size "${DECODE_BATCH_SIZE:-32}" \
    "${decode_resume[@]}"
}

pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
  run_one "${index}" "${GPU_IDS[${index}]}" >"${SHARD_ROOT}/logs/shard_${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${SHARD_ROOT}/logs/shard_"*.log >&2 || true
  exit "${status}"
fi

"${ENV_ROOT}/bin/python" -m evaluation.merge_evaluation_shards \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --shard-root "${SHARD_ROOT}" \
  --num-shards "${NUM_GPUS}" \
  --modes quality performance

verify_args=()
if [[ "${ALLOW_GENERATED_FAILURES:-0}" == "1" ]]; then
  verify_args+=(--allow-generated-failures)
fi
"${ENV_ROOT}/bin/python" "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_audio_output.py" \
  --manifest "${MANIFEST}" \
  --results "${OUTPUT_ROOT}/results.jsonl" \
  --summary "${OUTPUT_ROOT}/summary.json" \
  --expected-modes quality performance \
  "${verify_args[@]}"

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/vllm/generation_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/text_bleu.json"
"${ENV_ROOT}/bin/python" -m evaluation.slc_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics"
