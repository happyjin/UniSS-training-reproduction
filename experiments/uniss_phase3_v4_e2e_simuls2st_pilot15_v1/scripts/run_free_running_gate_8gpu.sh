#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?RUN_ID is required}"
: "${CANDIDATE_HF:?CANDIDATE_HF is required}"

FORMAL_DATA_RUN_ID=${FORMAL_DATA_RUN_ID:-formal_gold_20260818T090515Z}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/free_running_gates/${RUN_ID}}
SELECTION=${SELECTION:-${RUN_ROOT}/SELECTION.json}
CANDIDATE_FINGERPRINT=${CANDIDATE_FINGERPRINT:-${RUN_ROOT}/CANDIDATE_HF_FINGERPRINT.json}
GOLD=${GOLD:-${REPO_ROOT}/data/processed/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/source_events/valid_gold_trajectories.jsonl}
CANARY_REPORT=${CANARY_REPORT:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/post_task_pool_canaries/post_task_pool_canary_p4_replayfix_w0_20260821T085848Z/CANARY_REPORT.json}
CANDIDATE_CHECKPOINT=${CANDIDATE_CHECKPOINT:-${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}/post_task_pool_canaries/post_task_pool_canary_p4_replayfix_w0_20260821T085848Z/structural/iter_0000002}
BICODEC_MODEL=${BICODEC_MODEL:-${REPO_ROOT}/pretrained_models/UniSS/bicodec}
NUM_WORKERS=${NUM_WORKERS:-8}

required=(
  "${SELECTION}"
  "${CANDIDATE_FINGERPRINT}"
  "${GOLD}"
  "${GOLD}.offsets.bin"
  "${CANARY_REPORT}"
  "${CANDIDATE_CHECKPOINT}/metadata.json"
  "${CANDIDATE_HF}/model.safetensors"
  "${PHASE3_HF_MODEL}/model.safetensors"
  "${V1_CHECKPOINT}/metadata.json"
  "${WHISPERVQ_MODEL}/config.json"
  "${BICODEC_MODEL}/config.yaml"
)
for value in "${required[@]}"; do
  [[ -f "${value}" ]] || { echo "missing free-running gate input: ${value}" >&2; exit 2; }
done
[[ "${NUM_WORKERS}" == "8" ]] || { echo "formal free-running gate requires eight GPU workers" >&2; exit 2; }

GATE=${RUN_ROOT}/E2E_FREE_RUNNING_GATE.json
GPU_LOG=${RUN_ROOT}/gpu.csv
[[ ! -e "${GATE}" && ! -e "${GPU_LOG}" && ! -e "${RUN_ROOT}/workers" && ! -e "${RUN_ROOT}/audio" ]] || {
  echo "refusing to overwrite free-running gate run ${RUN_ID}" >&2
  exit 3
}

mapfile -t active_gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 != "[N/A]" {print $1}' | sort -u
)
if (( ${#active_gpu_pids[@]} > 0 )); then
  printf 'GPUs are busy; refusing to interfere with PIDs: %s\n' "${active_gpu_pids[*]}" >&2
  exit 4
fi

NVIDIA_LIBRARY_PATH=$(find "$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia" \
  -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${USER_ROOT}/.cache/pip"
export TMPDIR="${USER_ROOT}/tmp"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${RUN_ROOT}/workers" "${RUN_ROOT}/audio" "${RUN_ROOT}/logs" "${TMPDIR}"

CANDIDATE_SHA=$("${PYTHON_BIN}" -c '
import json,sys
x=json.load(open(sys.argv[1]))
v=x["checkpoints"]["candidate_hf"]
assert v["path"] == str(__import__("pathlib").Path(sys.argv[2]).resolve())
print(v["sha256"])
' "${CANDIDATE_FINGERPRINT}" "${CANDIDATE_HF}")
[[ "${#CANDIDATE_SHA}" == "64" ]] || { echo "malformed candidate HF fingerprint" >&2; exit 5; }

(
  echo "timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w,power_limit_w"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "${GPU_LOG}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

pids=()
for worker in $(seq 0 7); do
  report=${RUN_ROOT}/workers/worker_$(printf '%02d' "${worker}").json
  audio=${RUN_ROOT}/audio/worker_$(printf '%02d' "${worker}")
  log=${RUN_ROOT}/logs/worker_$(printf '%02d' "${worker}").log
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.run_worker \
    --selection "${SELECTION}" \
    --gold "${GOLD}" \
    --candidate-hf "${CANDIDATE_HF}" \
    --phase3-hf "${PHASE3_HF_MODEL}" \
    --v1-checkpoint "${V1_CHECKPOINT}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" \
    --candidate-hf-sha256 "${CANDIDATE_SHA}" \
    --worker-index "${worker}" \
    --num-workers "${NUM_WORKERS}" \
    --report "${report}" \
    --audio-dir "${audio}" \
    --device cuda:0 > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "one or more free-running workers failed; preserving immutable partial output" >&2
  exit 6
fi

worker_args=()
for worker in $(seq 0 7); do
  worker_args+=(--worker-report "${RUN_ROOT}/workers/worker_$(printf '%02d' "${worker}").json")
done
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.finalize_gate \
  --canary-report "${CANARY_REPORT}" \
  --selection "${SELECTION}" \
  "${worker_args[@]}" \
  --candidate-checkpoint "${CANDIDATE_CHECKPOINT}" \
  --candidate-hf "${CANDIDATE_HF}" \
  --v1-initialization "${V1_CHECKPOINT}" \
  --output "${GATE}" | tee "${RUN_ROOT}/FINALIZE.stdout.json"

