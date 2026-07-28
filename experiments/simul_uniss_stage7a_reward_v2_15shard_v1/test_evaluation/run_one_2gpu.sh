#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {r0_e3_v1_bias|r1_rebalanced_coverage|r2_explicit_latency|r3_bilingual_adaptive} [--smoke]" >&2
  exit 2
fi
LABEL="$1"
SMOKE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

case "${LABEL}" in
  r0_e3_v1_bias)
    MODEL="${R0_TEST_MODEL}"; GPU_LIST="${R0_TEST_GPUS}"
    BEST_STEP="${R0_BEST_STEP}"; WRITE_LOGIT_BIAS="${R0_WRITE_LOGIT_BIAS}"
    ;;
  r1_rebalanced_coverage)
    MODEL="${R1_TEST_MODEL}"; GPU_LIST="${R1_TEST_GPUS}"
    BEST_STEP="${R1_BEST_STEP}"; WRITE_LOGIT_BIAS="${R1_WRITE_LOGIT_BIAS}"
    ;;
  r2_explicit_latency)
    MODEL="${R2_TEST_MODEL}"; GPU_LIST="${R2_TEST_GPUS}"
    BEST_STEP="${R2_BEST_STEP}"; WRITE_LOGIT_BIAS="${R2_WRITE_LOGIT_BIAS}"
    ;;
  r3_bilingual_adaptive)
    MODEL="${R3_TEST_MODEL}"; GPU_LIST="${R3_TEST_GPUS}"
    BEST_STEP="${R3_BEST_STEP}"; WRITE_LOGIT_BIAS="${R3_WRITE_LOGIT_BIAS}"
    ;;
  *) echo "Unsupported Reward-v2 experiment: ${LABEL}" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || { echo "${LABEL} requires exactly two GPUs" >&2; exit 2; }
[[ -f "${MODEL}/config.json" && -f "${TEST_SCHEDULES}" ]] || {
  echo "Missing model or test schedules for ${LABEL}" >&2
  exit 1
}

LIMIT_RECORDS=0
EXPECTED="${EXPECTED_TEST_RECORDS}"
RUN_ID="${FULL_RUN_ID}"
RUN_ROOT="${TEST_EVAL_ROOT}"
RUN_LATENCY=1
if [[ "${SMOKE}" == "--smoke" ]]; then
  LIMIT_RECORDS="${SMOKE_RECORDS}"
  EXPECTED="${SMOKE_RECORDS}"
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
STREAMING_MODE="stage7a_reward_v2_${LABEL}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${RUN_DIR}/environment/git_status.txt"
cp "${SCRIPT_DIR}/experiment.env" "${RUN_DIR}/environment/experiment.env"
"${TRAIN_ENV}/bin/python" - "${RUN_DIR}/environment/run_manifest.json" \
  "${LABEL}" "${MODEL}" "${GPU_LIST}" "${EXPECTED}" "${BEST_STEP}" "${WRITE_LOGIT_BIAS}" <<'PY'
import json
import pathlib
import sys
import time

path, label, model, gpus, expected, step, bias = sys.argv[1:]
pathlib.Path(path).write_text(
    json.dumps(
        {
            "schema_version": "simul_uniss_stage7a_reward_v2_full_test_run_v1",
            "created_at_unix": time.time(),
            "experiment": label,
            "model": str(pathlib.Path(model).resolve()),
            "gpus": gpus,
            "expected_records": int(expected),
            "best_step": int(step),
            "write_logit_bias": float(bias),
            "selection_split": "dev",
            "evaluation_split": "test",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: test generation on GPUs ${GPU_LIST}"
pids=()
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_generate \
    --model "${MODEL}" --schedules "${TEST_SCHEDULES}" \
    --output-dir "${RUN_DIR}/generation" --rank "${rank}" --world-size 2 \
    --limit-records "${LIMIT_RECORDS}" --batch-records "${BATCH_RECORDS}" \
    --max-write-tokens "${MAX_WRITE_TOKENS}" --repetition-penalty "${REPETITION_PENALTY}" \
    --write-logit-bias "${WRITE_LOGIT_BIAS}" \
    --max-model-len "${MAX_MODEL_LEN}" --training-context-limit "${TRAINING_CONTEXT_LIMIT}" \
    --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --streaming-mode "${STREAMING_MODE}" --dtype bfloat16 --resume \
    > "${RUN_DIR}/logs/generation_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${RUN_DIR}/logs/generation_rank"*.log >&2 || true
  exit "${status}"
fi
if [[ ! -f "${RUN_DIR}/generation_results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/generation" --pattern 'generation.rank*.jsonl' \
    --output "${RUN_DIR}/generation_results.jsonl" \
    --expected-records "${EXPECTED}" --expected-ranks 2
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: BiCodec decode"
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
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${RUN_DIR}/logs/decode_rank"*.log >&2 || true
  exit "${status}"
fi
if [[ ! -f "${RUN_DIR}/results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/audio" --pattern 'results.rank*.jsonl' \
    --output "${RUN_DIR}/results.jsonl" \
    --expected-records "${EXPECTED}" --expected-ranks 2
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LABEL}: objective and streaming metrics"
mkdir -p "${RUN_DIR}/metrics"
if [[ ! -f "${RUN_DIR}/metrics/text_bleu.json" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.text_metrics \
    --input "${RUN_DIR}/generation_results.jsonl" --output "${RUN_DIR}/metrics/text_bleu.json"
fi
if [[ ! -f "${RUN_DIR}/metrics/slc.json" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.slc_metrics \
    --input "${RUN_DIR}/results.jsonl" --output-dir "${RUN_DIR}/metrics"
fi

METRIC_GPU_LIST="${GPU_LIST}"
for ((worker_copy = 1; worker_copy < METRIC_WORKERS_PER_GPU; worker_copy++)); do
  METRIC_GPU_LIST="${METRIC_GPU_LIST},${GPU_LIST}"
done
METRIC_NUM_WORKERS=$((2 * METRIC_WORKERS_PER_GPU))
EVAL_GPU_LIST="${METRIC_GPU_LIST}" METRIC_NUM_GPUS="${METRIC_NUM_WORKERS}" ENV_ROOT="${EVAL_ENV}" \
ASR_BATCH_SIZE="${ASR_BATCH_SIZE}" AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE}" \
AUTOPCP_CHUNK_SIZE="${AUTOPCP_CHUNK_SIZE}" \
  "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh" "${RUN_DIR}"

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_TEST}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/${LABEL}_streaming_test_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" --gpu-ids "${GPU_LIST}" \
  --split-label test --stage-label "${LABEL}" --stage-iteration "${BEST_STEP}" \
  --stage-description "Stage7A Reward-v2 frozen-dev operating point" \
  --streaming-mode "${STREAMING_MODE}" --expected-records "${EXPECTED}"

if [[ "${RUN_LATENCY}" -eq 1 ]]; then
  "${SCRIPT_DIR}/run_latency_2gpu.sh" \
    "${LABEL}" "${MODEL}" "${GPU_LIST}" "${RUN_DIR}" "${BEST_STEP}" "${WRITE_LOGIT_BIAS}"
fi

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"

