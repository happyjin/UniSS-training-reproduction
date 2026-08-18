#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

if [[ "${DATA_RUN_ID}" == "gold_smoke_v1" ]]; then
  echo "V1 teacher cache requires the explicit immutable gold DATA_RUN_ID" >&2
  exit 2
fi
: "${V1_ROLLOUT_RUN_ID:?set V1_ROLLOUT_RUN_ID to an audited rollout run}"

TEACHER_RUN_ID=${TEACHER_RUN_ID:-v1_teacher_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
TEACHER_SPLIT=${TEACHER_SPLIT:-valid}
TEACHER_START_INDEX=${TEACHER_START_INDEX:-0}
TEACHER_LIMIT=${TEACHER_LIMIT:-64}
NUM_GPUS=${NUM_GPUS:-8}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-1}
NUM_WORKERS=$((NUM_GPUS * PROCESSES_PER_GPU))
TOPK=${TOPK:-32}
TEMPERATURE=${TEMPERATURE:-1.5}
RECORDS_PER_BUNDLE=${RECORDS_PER_BUNDLE:-64}

if [[ "${TEACHER_SPLIT}" != "train" && "${TEACHER_SPLIT}" != "valid" ]]; then
  echo "TEACHER_SPLIT must be train or valid" >&2
  exit 2
fi
if (( NUM_GPUS < 1 || NUM_GPUS > 8 || PROCESSES_PER_GPU < 1 )); then
  echo "NUM_GPUS must be in [1,8] and PROCESSES_PER_GPU must be positive" >&2
  exit 2
fi
if (( TEACHER_LIMIT > 0 && TEACHER_LIMIT < NUM_WORKERS )); then
  echo "TEACHER_LIMIT must be zero/full or at least NUM_WORKERS" >&2
  exit 2
fi

GOLD=${PROCESSED_ROOT}/source_events/${TEACHER_SPLIT}_gold_trajectories.jsonl
ROLLOUT_ROOT=${PROCESSED_ROOT}/v1_rollouts/${V1_ROLLOUT_RUN_ID}
ROLLOUT=${ROLLOUT_ROOT}/${TEACHER_SPLIT}_v1_rollouts.jsonl
ROLLOUT_AUDIT=${REPORT_ROOT}/v1_rollouts/${V1_ROLLOUT_RUN_ID}/AUDIT.json
ROLLOUT_QUALITY=${REPORT_ROOT}/v1_rollouts/${V1_ROLLOUT_RUN_ID}/QUALITY_GATE.json
TEACHER_ROOT=${PROCESSED_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}
TEACHER_REPORT_ROOT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}
TEACHER_LOG_ROOT=${LOG_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}
PARTS_ROOT=${TEACHER_ROOT}/parts
MERGED=${TEACHER_ROOT}/${TEACHER_SPLIT}_v1_asr_teacher_cache.jsonl
AUDIT=${TEACHER_REPORT_ROOT}/AUDIT.json
GPU_SUMMARY=${TEACHER_REPORT_ROOT}/GPU_SUMMARY.json
HF_FINGERPRINT=${TEACHER_REPORT_ROOT}/V1_HF_FINGERPRINT.json

for path in "${TEACHER_ROOT}" "${TEACHER_REPORT_ROOT}" "${TEACHER_LOG_ROOT}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite V1 teacher-cache run: ${path}" >&2
    exit 2
  fi
done
for path in "${GOLD}" "${ROLLOUT}" "${ROLLOUT_AUDIT}" "${ROLLOUT_QUALITY}" "${V1_CHECKPOINT}" "${V1_HF_MODEL}" "${WHISPERVQ_MODEL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required V1 teacher-cache input is missing: ${path}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - <<'PY' "${ROLLOUT_AUDIT}" "${ROLLOUT_QUALITY}" "${ROLLOUT}" "${GOLD}"
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
quality = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if audit.get("status") != "passed" or quality.get("status") != "passed":
    raise SystemExit("V1 rollout audit or quality gate did not pass")
for value in (audit, quality):
    if pathlib.Path(str(value.get("rollouts", ""))).resolve() != pathlib.Path(sys.argv[3]).resolve():
        raise SystemExit("V1 rollout gate points to a different rollout")
    if pathlib.Path(str(value.get("gold", ""))).resolve() != pathlib.Path(sys.argv[4]).resolve():
        raise SystemExit("V1 rollout gate points to a different gold file")
PY

mkdir -p "${PARTS_ROOT}" "${TEACHER_REPORT_ROOT}" "${TEACHER_LOG_ROOT}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/.cache/torch/kernels}
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

"${PYTHON_BIN}" -m experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1_hf=${V1_HF_MODEL}" \
  --output "${HF_FINGERPRINT}" \
  --workers 16 \
  > "${TEACHER_LOG_ROOT}/hf_fingerprint.log" 2>&1
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
  > "${TEACHER_LOG_ROOT}/gpu_dmon.log" 2>&1 &
MONITOR_PID=$!

pids=()
for ((worker=0; worker<NUM_WORKERS; worker++)); do
  gpu=$((worker % NUM_GPUS))
  part=$(printf '%s/part_%03d' "${PARTS_ROOT}" "${worker}")
  log=$(printf '%s/rank%03d.log' "${TEACHER_LOG_ROOT}" "${worker}")
  limit_args=()
  if (( TEACHER_LIMIT > 0 )); then
    limit_args=(--limit "${TEACHER_LIMIT}")
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.build_v1_cache \
    --gold "${GOLD}" \
    --rollouts "${ROLLOUT}" \
    --output-dir "${part}" \
    --checkpoint "${V1_CHECKPOINT}" \
    --hf-model "${V1_HF_MODEL}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --v1-hf-sha256 "${V1_HF_SHA}" \
    --rank "${worker}" \
    --world-size "${NUM_WORKERS}" \
    --start-index "${TEACHER_START_INDEX}" \
    --topk "${TOPK}" \
    --temperature "${TEMPERATURE}" \
    --records-per-bundle "${RECORDS_PER_BUNDLE}" \
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
  echo "V1 teacher-cache worker failed; inspect ${TEACHER_LOG_ROOT}" >&2
  exit "${status}"
fi
cleanup
MONITOR_PID=""

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.summarize_gpu_dmon \
  --input "${TEACHER_LOG_ROOT}/gpu_dmon.log" \
  --output "${GPU_SUMMARY}" \
  --minimum-active-memory-mib 512 \
  > "${TEACHER_LOG_ROOT}/gpu_summary.log" 2>&1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.merge_v1_cache \
  --gold "${GOLD}" \
  --parts-root "${PARTS_ROOT}" \
  --world-size "${NUM_WORKERS}" \
  --output "${MERGED}" \
  --audit "${AUDIT}" \
  > "${TEACHER_LOG_ROOT}/merge_audit.log" 2>&1

echo "teacher_cache=${MERGED}"
echo "audit=${AUDIT}"
echo "gpu_summary=${GPU_SUMMARY}"
echo "gpu_monitor=${TEACHER_LOG_ROOT}/gpu_dmon.log"
