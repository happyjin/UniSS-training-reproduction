#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${LEARNING_RUN_ID:?LEARNING_RUN_ID is required}"
: "${LEARNING_ITER:?LEARNING_ITER is required}"
: "${CANDIDATE_HF:?CANDIDATE_HF is required}"
: "${GATE_RUN_ID:?GATE_RUN_ID is required}"

FORMAL_DATA_RUN_ID=${FORMAL_DATA_RUN_ID:-formal_gold_20260818T090515Z}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-64}
EXPORT_ATTEMPT_ID=${EXPORT_ATTEMPT_ID:-${LEARNING_RUN_ID}}
MEGATRON_CHECKPOINT=${CHECKPOINT_ROOT}/learning_canaries/${LEARNING_RUN_ID}/iter_$(printf '%07d' "$((10#${LEARNING_ITER}))")
GATE_ROOT=${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/free_running_gates/${GATE_RUN_ID}
SELECTION=${SELECTION:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/free_running_gates/free_running_gate_learning100u_20260821T142900Z/SELECTION.json}
EXPORT_LOG=${LOG_ROOT}/e2e_${EXPORT_ATTEMPT_ID}_export_after_gpu_restore.log
WATCH_LOG=${LOG_ROOT}/e2e_${EXPORT_ATTEMPT_ID}_gpu_restore_wait.log

required=(
  "${MEGATRON_CHECKPOINT}/metadata.json"
  "${SELECTION}"
  "${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf/model.safetensors"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing GPU-restore gate input: ${path}" >&2; exit 2; }
done
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${POLL_SECONDS}" -ge 10 ]] || {
  echo "POLL_SECONDS must be an integer of at least 10" >&2
  exit 2
}
[[ ! -e "${WATCH_LOG}" ]] || { echo "refusing to overwrite ${WATCH_LOG}" >&2; exit 3; }
[[ ! -e "${GATE_ROOT}" ]] || { echo "refusing to overwrite ${GATE_ROOT}" >&2; exit 3; }
[[ ! -e "${CANDIDATE_HF}" ]] || { echo "refusing to overwrite ${CANDIDATE_HF}" >&2; exit 3; }

mkdir -p "$(dirname -- "${WATCH_LOG}")"
exec > >(tee -a "${WATCH_LOG}") 2>&1
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "waiting_for=visible_and_idle_8gpu"

while true; do
  visible=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l || true)
  if [[ "${visible}" == "8" ]]; then
    mapfile -t active_gpu_pids < <(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "[N/A]" {print $1}' | sort -u
    )
    if (( ${#active_gpu_pids[@]} == 0 )); then
      break
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs visible but busy: ${active_gpu_pids[*]}"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) visible_gpus=${visible}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "gpu_ready_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
NVIDIA_LIBRARY_ROOT="$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia"
NVIDIA_LIBRARY_PATH=
if [[ -d "${NVIDIA_LIBRARY_ROOT}" ]]; then
  NVIDIA_LIBRARY_PATH=$(find "${NVIDIA_LIBRARY_ROOT}" \
    -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
fi
SYSTEM_CUDA_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${SYSTEM_CUDA_LIBRARY_PATH}:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}${NVIDIA_LIBRARY_PATH:+:${NVIDIA_LIBRARY_PATH}}"
"${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf" \
  --megatron-path "${MEGATRON_CHECKPOINT}" \
  --hf-output "${CANDIDATE_HF}" \
  --model-type gpt 2>&1 | tee "${EXPORT_LOG}"

mkdir -p "${GATE_ROOT}"
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "candidate_hf=${CANDIDATE_HF}" \
  --workers 12 \
  --output "${GATE_ROOT}/CANDIDATE_HF_FINGERPRINT.json" \
  | tee "${GATE_ROOT}/FINGERPRINT.stdout.json"

env \
  RUN_ID="${GATE_RUN_ID}" \
  RUN_ROOT="${GATE_ROOT}" \
  SELECTION="${SELECTION}" \
  CANDIDATE_HF="${CANDIDATE_HF}" \
  CANDIDATE_FINGERPRINT="${GATE_ROOT}/CANDIDATE_HF_FINGERPRINT.json" \
  CANDIDATE_CHECKPOINT="${MEGATRON_CHECKPOINT}" \
  NUM_WORKERS=8 \
  MAX_S2S_SEMANTIC_TOKENS="${MAX_S2S_SEMANTIC_TOKENS}" \
  "${SCRIPT_DIR}/run_free_running_gate_8gpu.sh"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "gate=${GATE_ROOT}/E2E_FREE_RUNNING_GATE.json"
