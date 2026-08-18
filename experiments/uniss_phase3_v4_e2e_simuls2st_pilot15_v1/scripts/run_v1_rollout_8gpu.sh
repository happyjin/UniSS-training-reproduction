#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

if [[ "${DATA_RUN_ID}" == "gold_smoke_v1" ]]; then
  echo "V1 rollout requires the explicit immutable gold DATA_RUN_ID" >&2
  exit 2
fi
ROLLOUT_RUN_ID=${ROLLOUT_RUN_ID:-v1_rollout_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
ROLLOUT_SPLIT=${ROLLOUT_SPLIT:-valid}
ROLLOUT_LIMIT=${ROLLOUT_LIMIT:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_EVENT_TOKENS=${MAX_EVENT_TOKENS:-96}
MAX_FINAL_TOKENS=${MAX_FINAL_TOKENS:-8}

if [[ "${ROLLOUT_SPLIT}" != "train" && "${ROLLOUT_SPLIT}" != "valid" ]]; then
  echo "ROLLOUT_SPLIT must be train or valid" >&2
  exit 2
fi
if (( NUM_WORKERS < 1 || NUM_WORKERS > 8 )); then
  echo "NUM_WORKERS must be in [1,8] for the one-process-per-GPU launcher" >&2
  exit 2
fi
if (( ROLLOUT_LIMIT > 0 && ROLLOUT_LIMIT < NUM_WORKERS )); then
  echo "ROLLOUT_LIMIT must be zero/full or at least NUM_WORKERS" >&2
  exit 2
fi

GOLD=${PROCESSED_ROOT}/source_events/${ROLLOUT_SPLIT}_gold_trajectories.jsonl
ROLLOUT_ROOT=${PROCESSED_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
ROLLOUT_REPORT_ROOT=${REPORT_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
ROLLOUT_LOG_ROOT=${LOG_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
PART_ROOT=${ROLLOUT_ROOT}/parts
MERGED=${ROLLOUT_ROOT}/${ROLLOUT_SPLIT}_v1_rollouts.jsonl
MERGE_REPORT=${ROLLOUT_REPORT_ROOT}/MERGE.json
AUDIT_JSON=${ROLLOUT_REPORT_ROOT}/AUDIT.json
AUDIT_MD=${ROLLOUT_REPORT_ROOT}/AUDIT.md
HF_FINGERPRINT=${ROLLOUT_REPORT_ROOT}/V1_HF_FINGERPRINT.json

for path in "${ROLLOUT_ROOT}" "${ROLLOUT_REPORT_ROOT}" "${ROLLOUT_LOG_ROOT}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite rollout run: ${path}" >&2
    exit 2
  fi
done
for path in "${GOLD}" "${V1_CHECKPOINT}" "${V1_HF_MODEL}" "${WHISPERVQ_MODEL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required rollout input is missing: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${PART_ROOT}" "${ROLLOUT_REPORT_ROOT}/workers" "${ROLLOUT_LOG_ROOT}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/.cache/torch/kernels}
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

"${PYTHON_BIN}" -m experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1_hf=${V1_HF_MODEL}" \
  --output "${HF_FINGERPRINT}" \
  --workers 16 \
  > "${ROLLOUT_LOG_ROOT}/hf_fingerprint.log" 2>&1
V1_HF_SHA=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["v1_hf"]["sha256"])' "${HF_FINGERPRINT}")

MONITOR_PID=""
cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi dmon -s pucvmet -d 2 -o DT \
  > "${ROLLOUT_LOG_ROOT}/gpu_dmon.log" 2>&1 &
MONITOR_PID=$!

pids=()
reports=()
for ((worker=0; worker<NUM_WORKERS; worker++)); do
  part=$(printf '%s/rank%03d.jsonl' "${PART_ROOT}" "${worker}")
  report=$(printf '%s/workers/rank%03d.json' "${ROLLOUT_REPORT_ROOT}" "${worker}")
  log=$(printf '%s/rank%03d.log' "${ROLLOUT_LOG_ROOT}" "${worker}")
  reports+=("${report}")
  limit_args=()
  if (( ROLLOUT_LIMIT > 0 )); then
    limit_args=(--limit "${ROLLOUT_LIMIT}")
  fi
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.run_worker \
    --input "${GOLD}" \
    --output "${part}" \
    --report "${report}" \
    --checkpoint "${V1_CHECKPOINT}" \
    --hf-model "${V1_HF_MODEL}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --v1-hf-sha256 "${V1_HF_SHA}" \
    --worker-index "${worker}" \
    --num-workers "${NUM_WORKERS}" \
    --max-event-tokens "${MAX_EVENT_TOKENS}" \
    --max-final-tokens "${MAX_FINAL_TOKENS}" \
    --device cuda:0 \
    "${limit_args[@]}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if (( status != 0 )); then
  echo "V1 rollout worker failed; inspect ${ROLLOUT_LOG_ROOT}" >&2
  exit "${status}"
fi
cleanup
MONITOR_PID=""

merge_args=()
for report in "${reports[@]}"; do
  merge_args+=(--part-report "${report}")
done
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.merge_parts \
  "${merge_args[@]}" \
  --output "${MERGED}" \
  --report "${MERGE_REPORT}" \
  > "${ROLLOUT_LOG_ROOT}/merge.log" 2>&1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.audit_rollouts \
  --gold "${GOLD}" \
  --rollouts "${MERGED}" \
  --merge-report "${MERGE_REPORT}" \
  --output-json "${AUDIT_JSON}" \
  --output-md "${AUDIT_MD}" \
  > "${ROLLOUT_LOG_ROOT}/audit.log" 2>&1

echo "rollout=${MERGED}"
echo "audit=${AUDIT_JSON}"
echo "report=${AUDIT_MD}"
echo "gpu_monitor=${ROLLOUT_LOG_ROOT}/gpu_dmon.log"
