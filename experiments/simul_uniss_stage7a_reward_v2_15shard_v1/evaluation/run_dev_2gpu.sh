#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 LABEL MODEL GPU_LIST WRITE_LOGIT_BIAS [--smoke]" >&2
  exit 2
fi
LABEL="$1"
MODEL="$2"
GPU_LIST="$3"
WRITE_LOGIT_BIAS="$4"
SMOKE="${5:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || { echo "Exactly two GPUs are required" >&2; exit 2; }
[[ -f "${MODEL}/config.json" && -f "${DEV_SCHEDULES}" ]] || { echo "Missing model or dev schedules" >&2; exit 1; }

LIMIT_RECORDS=0
EXPECTED_RECORDS="${EXPECTED_DEV_RECORDS}"
BATCH_RECORDS="${GEN_BATCH_RECORDS}"
MAX_SEQS_LOCAL="${MAX_NUM_SEQS}"
MAX_BATCHED_TOKENS_LOCAL="${MAX_NUM_BATCHED_TOKENS}"
MEMORY_UTILIZATION_LOCAL="${GPU_MEMORY_UTILIZATION}"
RUN_DIR="${EVAL_ROOT}/full_dev_e2e_v1/${LABEL}"
if [[ "${SMOKE}" == "--smoke" ]]; then
  LIMIT_RECORDS=16
  EXPECTED_RECORDS=16
  BATCH_RECORDS=16
  MAX_SEQS_LOCAL=16
  MAX_BATCHED_TOKENS_LOCAL=32768
  MEMORY_UTILIZATION_LOCAL=0.5
  RUN_DIR="${EVAL_ROOT}/smoke_dev_e2e/${LABEL}_$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ -n "${SMOKE}" ]]; then
  echo "Unknown argument: ${SMOKE}" >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" && "${RESUME:-0}" != 1 ]]; then
  echo "Refusing to overwrite ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/environment" "${RUN_DIR}/logs" "${RUN_DIR}/generation"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${RUN_DIR}/environment/git_status.txt"
cp "${ROOT}/experiment.env" "${RUN_DIR}/environment/experiment.env"

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: free-running generation"
pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_generate \
    --model "${MODEL}" --schedules "${DEV_SCHEDULES}" \
    --output-dir "${RUN_DIR}/generation" --rank "${rank}" --world-size 2 \
    --limit-records "${LIMIT_RECORDS}" --batch-records "${BATCH_RECORDS}" \
    --max-write-tokens "${MAX_WRITE_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" --training-context-limit "${MAX_SEQUENCE_LENGTH}" \
    --max-num-seqs "${MAX_SEQS_LOCAL}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS_LOCAL}" \
    --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}" \
    --gpu-memory-utilization "${MEMORY_UTILIZATION_LOCAL}" \
    --repetition-penalty 1.1 --write-logit-bias "${WRITE_LOGIT_BIAS}" \
    --streaming-mode "stage7a_reward_v2_${LABEL}" --dtype bfloat16 --resume \
    > "${RUN_DIR}/logs/generation_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then tail -n 100 "${RUN_DIR}/logs/generation_rank"*.log >&2 || true; exit "${status}"; fi
if [[ ! -f "${RUN_DIR}/generation_results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/generation" --pattern 'generation.rank*.jsonl' \
    --output "${RUN_DIR}/generation_results.jsonl" \
    --expected-records "${EXPECTED_RECORDS}" --expected-ranks 2
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: BiCodec decode"
mkdir -p "${RUN_DIR}/audio"
pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${TRAIN_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_decode \
    --input "${RUN_DIR}/generation_results.jsonl" --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${RUN_DIR}/audio" --rank "${rank}" --world-size 2 --device cuda:0 \
    --batch-size "${DECODE_BATCH_SIZE}" --left-context-tokens 50 \
    --holdback-tokens 5 --overlap-ms 80 \
    --artifact-prefix "stage7a_reward_v2_${LABEL}" --resume \
    > "${RUN_DIR}/logs/decode_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then tail -n 100 "${RUN_DIR}/logs/decode_rank"*.log >&2 || true; exit "${status}"; fi
if [[ ! -f "${RUN_DIR}/results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/audio" --pattern 'results.rank*.jsonl' \
    --output "${RUN_DIR}/results.jsonl" \
    --expected-records "${EXPECTED_RECORDS}" --expected-ranks 2
fi

mkdir -p "${RUN_DIR}/metrics"
"${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
  --input "${RUN_DIR}/generation_results.jsonl" --output "${RUN_DIR}/metrics/text_bleu.json"
"${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
  --input "${RUN_DIR}/results.jsonl" --output-dir "${RUN_DIR}/metrics"

METRIC_GPU_LIST="${GPU_LIST},${GPU_LIST}"
EVAL_GPU_LIST="${METRIC_GPU_LIST}" METRIC_NUM_GPUS=4 ENV_ROOT="${EVAL_ENV}" \
ASR_BATCH_SIZE="${ASR_BATCH_SIZE}" AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE}" \
AUTOPCP_CHUNK_SIZE="${AUTOPCP_CHUNK_SIZE}" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh" "${RUN_DIR}"

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_DEV}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/${LABEL}_full_dev_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" --gpu-ids "${GPU_LIST}" \
  --split-label dev --stage-label "${LABEL}" --stage-iteration 0 \
  --stage-description "Stage7A Reward-v2 15-shard dev evaluation" \
  --streaming-mode "stage7a_reward_v2_${LABEL}" \
  --expected-records "${EXPECTED_RECORDS}"

cleanup; trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"
