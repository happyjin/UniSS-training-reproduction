#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_DIR LIMIT_RECORDS_OR_0" >&2
  exit 2
fi
RUN_DIR="$1"
LIMIT_RECORDS="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

IFS=',' read -r -a GPU_IDS <<<"${DEV_GPUS}"
[[ "${#GPU_IDS[@]}" -eq 4 ]] || { echo "DEV_GPUS must contain four GPU indexes" >&2; exit 2; }
mkdir -p "${RUN_DIR}/generation" "${RUN_DIR}/logs"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export GLOO_SOCKET_IFNAME=lo

pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_generate \
    --model "${HF_EXPORT}" \
    --schedules "${DEV_SCHEDULES}" \
    --output-dir "${RUN_DIR}/generation" \
    --rank "${rank}" --world-size 4 \
    --limit-records "${LIMIT_RECORDS}" \
    --batch-records "${BATCH_RECORDS}" \
    --max-write-tokens "${MAX_WRITE_TOKENS}" \
    --repetition-penalty "${REPETITION_PENALTY}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --training-context-limit "${TRAINING_CONTEXT_LIMIT}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype bfloat16 \
    --resume \
    >"${RUN_DIR}/logs/generation_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${RUN_DIR}/logs/generation_rank"*.log >&2 || true
  exit "${status}"
fi

expected="${EXPECTED_DEV_RECORDS}"
if [[ "${LIMIT_RECORDS}" != "0" ]]; then expected="${LIMIT_RECORDS}"; fi
if [[ ! -f "${RUN_DIR}/generation_results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/generation" \
    --pattern 'generation.rank*.jsonl' \
    --output "${RUN_DIR}/generation_results.jsonl" \
    --expected-records "${expected}" \
    --expected-ranks 4
fi
