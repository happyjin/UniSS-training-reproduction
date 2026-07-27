#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {e0_stage6|e1_continued_sft|e2_grpo_g4|e3_grpo_g8} [--smoke]" >&2
  exit 2
fi
LABEL="$1"
SMOKE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

case "${LABEL}" in
  e0_stage6) MODEL="${E0_MODEL}"; GPU_LIST="${E0_TEST_GPUS}"; BEST_STEP=1189 ;;
  e1_continued_sft) MODEL="${E1_MODEL}"; GPU_LIST="${E1_TEST_GPUS}"; BEST_STEP=700 ;;
  e2_grpo_g4) MODEL="${E2_MODEL}"; GPU_LIST="${E2_TEST_GPUS}"; BEST_STEP=600 ;;
  e3_grpo_g8) MODEL="${E3_MODEL}"; GPU_LIST="${E3_TEST_GPUS}"; BEST_STEP=700 ;;
  *) echo "Unsupported experiment: ${LABEL}" >&2; exit 2 ;;
esac
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || { echo "${LABEL} requires exactly two GPUs" >&2; exit 2; }
[[ -f "${MODEL}/config.json" && -f "${TEST_SCHEDULES}" ]] || { echo "Missing model or test schedules" >&2; exit 1; }

LIMIT_RECORDS=0
RUN_ID="${FULL_RUN_ID}"
RUN_ROOT="${TEST_EVAL_ROOT}"
RUN_OBJECTIVE_METRICS=1
RUN_LATENCY=1
if [[ "${SMOKE}" == "--smoke" ]]; then
  LIMIT_RECORDS="${SMOKE_RECORDS}"
  RUN_ID="smoke_${SMOKE_RECORDS}_$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="${TEST_EVAL_ROOT}/smoke"
  RUN_LATENCY=0
  BATCH_RECORDS="${SMOKE_RECORDS}"
  MAX_NUM_SEQS="${SMOKE_RECORDS}"
  MAX_NUM_BATCHED_TOKENS=32768
  GPU_MEMORY_UTILIZATION=0.50
  DECODE_BATCH_SIZE="${SMOKE_RECORDS}"
elif [[ -n "${SMOKE}" ]]; then
  echo "Unknown argument: ${SMOKE}" >&2
  exit 2
fi
RUN_DIR="${RUN_ROOT}/${LABEL}/${RUN_ID}"
if [[ -e "${RUN_DIR}" && "${RESUME:-0}" != 1 ]]; then
  echo "Refusing to overwrite ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/environment" "${RUN_DIR}/logs" "${RUN_DIR}/generation"

EXPECTED="${EXPECTED_TEST_RECORDS}"
if [[ "${LIMIT_RECORDS}" != 0 ]]; then EXPECTED="${LIMIT_RECORDS}"; fi
STREAMING_MODE="stage7a_${LABEL}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${RUN_DIR}/environment/git_status.txt"
cp "${SCRIPT_DIR}/experiment.env" "${RUN_DIR}/environment/experiment.env"
"${TRAIN_ENV}/bin/python" - "${RUN_DIR}/environment/run_manifest.json" "${LABEL}" "${MODEL}" "${GPU_LIST}" "${EXPECTED}" "${BEST_STEP}" <<'PY'
import json, pathlib, sys, time
path, label, model, gpus, expected, step = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "schema_version": "simul_uniss_stage7a_full_test_run_v1",
    "created_at_unix": time.time(), "experiment": label,
    "model": str(pathlib.Path(model).resolve()), "gpus": gpus,
    "expected_records": int(expected), "best_step": int(step),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: generation on GPUs ${GPU_LIST}"
pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_generate \
    --model "${MODEL}" --schedules "${TEST_SCHEDULES}" \
    --output-dir "${RUN_DIR}/generation" --rank "${rank}" --world-size 2 \
    --limit-records "${LIMIT_RECORDS}" --batch-records "${BATCH_RECORDS}" \
    --max-write-tokens "${MAX_WRITE_TOKENS}" --repetition-penalty "${REPETITION_PENALTY}" \
    --max-model-len "${MAX_MODEL_LEN}" --training-context-limit "${TRAINING_CONTEXT_LIMIT}" \
    --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --streaming-mode "${STREAMING_MODE}" --dtype bfloat16 --resume \
    > "${RUN_DIR}/logs/generation_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then tail -n 100 "${RUN_DIR}/logs/generation_rank"*.log >&2 || true; exit "${status}"; fi
"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
  --input-dir "${RUN_DIR}/generation" --pattern 'generation.rank*.jsonl' \
  --output "${RUN_DIR}/generation_results.jsonl" --expected-records "${EXPECTED}" --expected-ranks 2

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: audio decode"
mkdir -p "${RUN_DIR}/audio"
pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${TRAIN_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_decode \
    --input "${RUN_DIR}/generation_results.jsonl" --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${RUN_DIR}/audio" --rank "${rank}" --world-size 2 --device cuda:0 \
    --batch-size "${DECODE_BATCH_SIZE}" --left-context-tokens "${LEFT_CONTEXT_TOKENS}" \
    --holdback-tokens "${HOLDBACK_TOKENS}" --overlap-ms "${OVERLAP_MS}" \
    --artifact-prefix "${STREAMING_MODE}" --resume \
    > "${RUN_DIR}/logs/decode_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then tail -n 100 "${RUN_DIR}/logs/decode_rank"*.log >&2 || true; exit "${status}"; fi
"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
  --input-dir "${RUN_DIR}/audio" --pattern 'results.rank*.jsonl' \
  --output "${RUN_DIR}/results.jsonl" --expected-records "${EXPECTED}" --expected-ranks 2

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: quality and streaming metrics"
mkdir -p "${RUN_DIR}/metrics"
"${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
  --input "${RUN_DIR}/generation_results.jsonl" --output "${RUN_DIR}/metrics/text_bleu.json"
"${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
  --input "${RUN_DIR}/results.jsonl" --output-dir "${RUN_DIR}/metrics"
if [[ "${RUN_OBJECTIVE_METRICS}" == 1 ]]; then
  EVAL_GPU_LIST="${GPU_LIST}" METRIC_NUM_GPUS=2 ENV_ROOT="${EVAL_ENV}" \
  ASR_BATCH_SIZE="${ASR_BATCH_SIZE}" AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE}" \
  AUTOPCP_CHUNK_SIZE="${AUTOPCP_CHUNK_SIZE}" \
    "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh" "${RUN_DIR}"
fi
"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_TEST}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/${LABEL}_streaming_test_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" --gpu-ids "${GPU_LIST}" \
  --split-label test --stage-label "${LABEL}" --stage-iteration "${BEST_STEP}" \
  --stage-description "Stage7A isolated action-policy comparison" \
  --streaming-mode "${STREAMING_MODE}" --expected-records "${EXPECTED}"

if [[ "${RUN_LATENCY}" == 1 ]]; then
  "${SCRIPT_DIR}/run_latency_2gpu.sh" "${LABEL}" "${MODEL}" "${GPU_LIST}" "${RUN_DIR}" "${BEST_STEP}"
fi

cleanup; trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"
