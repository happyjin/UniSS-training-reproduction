#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"

RUN_DIR="${EVAL_ROOT}/full_dev_e2e_v1/r3_bilingual_adaptive"
GPU_LIST="${R3_AUTOPCP_RECOVERY_GPUS:-6,7}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || {
  echo "R3_AUTOPCP_RECOVERY_GPUS must contain exactly two GPUs" >&2
  exit 2
}
[[ -f "${RUN_DIR}/results.jsonl" ]] || {
  echo "Missing R3 dev results: ${RUN_DIR}/results.jsonl" >&2
  exit 1
}
[[ ! -f "${RUN_DIR}/COMPLETE" ]] || {
  echo "R3 dev evaluation is already complete"
  exit 0
}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${USER_ROOT}/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${USER_ROOT}/cache/torch}"

SHARD_ROOT="${RUN_DIR}/metrics/shards"
mkdir -p "${SHARD_ROOT}/logs"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 1 > "${RUN_DIR}/gpu_monitor_autopcp_resume.csv" &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

pids=()
for offset in 0 1; do
  shard_index=$((offset + 1))
  part_dir="${SHARD_ROOT}/autopcp/part_$(printf '%03d' "${shard_index}")"
  mkdir -p "${part_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${offset}]}" "${EVAL_ENV}/bin/python" \
    -m evaluation.autopcp_metrics \
    --input "${RUN_DIR}/results.jsonl" \
    --output-dir "${part_dir}" \
    --comparator-path "${USER_ROOT}/evaluation_models/AutoPCP-multilingual-v2" \
    --device cuda:0 \
    --pick-layer 9 \
    --symmetrize \
    --batch-size "${R3_AUTOPCP_RECOVERY_BATCH_SIZE:-16}" \
    --chunk-size "${R3_AUTOPCP_RECOVERY_CHUNK_SIZE:-512}" \
    --num-process 1 \
    --num-shards 4 \
    --shard-index "${shard_index}" \
    --resume \
    >"${SHARD_ROOT}/logs/autopcp_resume_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${SHARD_ROOT}/logs/autopcp_resume_"*.log >&2 || true
  exit "${status}"
fi

"${EVAL_ENV}/bin/python" -m evaluation.merge_metric_shards \
  --metric autopcp \
  --input "${RUN_DIR}/results.jsonl" \
  --metric-dir "${RUN_DIR}/metrics" \
  --shard-root "${SHARD_ROOT}" \
  --num-shards 4

"${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate report \
  --run-dir "${RUN_DIR}" \
  --results "${RUN_DIR}/results.jsonl" \
  --offline-phase3-root "${OFFLINE_PHASE3_DEV}" \
  --output-json "${RUN_DIR}/aggregate_metrics.json" \
  --report "${RUN_DIR}/r3_bilingual_adaptive_full_dev_report.md" \
  --gpu-monitor "${RUN_DIR}/gpu_monitor.csv" \
  --gpu-ids "${R3_GPUS}" \
  --split-label dev \
  --stage-label r3_bilingual_adaptive \
  --stage-iteration 0 \
  --stage-description "Stage7A Reward-v2 15-shard dev evaluation" \
  --streaming-mode stage7a_reward_v2_r3_bilingual_adaptive \
  --expected-records "${EXPECTED_DEV_RECORDS}"

cleanup
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
echo "RUN_DIR=${RUN_DIR}"
