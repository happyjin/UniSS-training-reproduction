#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 LABEL MODEL GPU_LIST PARENT_RUN_DIR BEST_STEP WRITE_LOGIT_BIAS" >&2
  exit 2
fi
LABEL="$1"
MODEL="$2"
GPU_LIST="$3"
PARENT_RUN_DIR="$4"
BEST_STEP="$5"
WRITE_LOGIT_BIAS="$6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || { echo "Exactly two GPUs are required" >&2; exit 2; }

RUN_DIR="${PARENT_RUN_DIR}/latency_batch1"
mkdir -p "${RUN_DIR}/generation" "${RUN_DIR}/audio" "${RUN_DIR}/logs" "${RUN_DIR}/metrics"
STREAMING_MODE="stage7a_reward_v2_${LABEL}_latency_batch1"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_generate \
    --model "${MODEL}" --schedules "${TEST_SCHEDULES}" \
    --output-dir "${RUN_DIR}/generation" --rank "${rank}" --world-size 2 \
    --limit-records "${LATENCY_RECORDS}" --batch-records 1 \
    --max-write-tokens "${MAX_WRITE_TOKENS}" --repetition-penalty "${REPETITION_PENALTY}" \
    --write-logit-bias "${WRITE_LOGIT_BIAS}" \
    --max-model-len "${MAX_MODEL_LEN}" --training-context-limit "${TRAINING_CONTEXT_LIMIT}" \
    --max-num-seqs 1 --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    --max-seq-len-to-capture 4096 --gpu-memory-utilization 0.50 \
    --streaming-mode "${STREAMING_MODE}" --dtype bfloat16 --resume \
    > "${RUN_DIR}/logs/generation_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
[[ "${status}" -eq 0 ]] || { tail -n 100 "${RUN_DIR}/logs/generation_rank"*.log >&2 || true; exit "${status}"; }

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
  --input-dir "${RUN_DIR}/generation" --pattern 'generation.rank*.jsonl' \
  --output "${RUN_DIR}/generation_results.jsonl" \
  --expected-records "${LATENCY_RECORDS}" --expected-ranks 2

pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${TRAIN_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_decode \
    --input "${RUN_DIR}/generation_results.jsonl" --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${RUN_DIR}/audio" --rank "${rank}" --world-size 2 --device cuda:0 \
    --batch-size 1 --left-context-tokens "${LEFT_CONTEXT_TOKENS}" \
    --holdback-tokens "${HOLDBACK_TOKENS}" --overlap-ms "${OVERLAP_MS}" \
    --artifact-prefix "${STREAMING_MODE}" --resume \
    > "${RUN_DIR}/logs/decode_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
[[ "${status}" -eq 0 ]] || { tail -n 100 "${RUN_DIR}/logs/decode_rank"*.log >&2 || true; exit "${status}"; }

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
  --input-dir "${RUN_DIR}/audio" --pattern 'results.rank*.jsonl' \
  --output "${RUN_DIR}/results.jsonl" \
  --expected-records "${LATENCY_RECORDS}" --expected-ranks 2
"${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
  --input "${RUN_DIR}/generation_results.jsonl" --output "${RUN_DIR}/metrics/text_bleu.json"
"${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
  --input "${RUN_DIR}/results.jsonl" --output-dir "${RUN_DIR}/metrics"
"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_TEST}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/${LABEL}_latency_batch1_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" --gpu-ids "${GPU_LIST}" \
  --split-label test-latency-batch1 --stage-label "${LABEL}" --stage-iteration "${BEST_STEP}" \
  --stage-description "Stage7A Reward-v2 batch-one latency audit" \
  --streaming-mode "${STREAMING_MODE}" --expected-records "${LATENCY_RECORDS}"

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"

