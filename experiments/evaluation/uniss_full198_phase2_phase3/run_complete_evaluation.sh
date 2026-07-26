#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3"
MANIFEST_ROOT="${EVAL_ROOT}/manifests"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-60}"
# Four independent H200 workers leave ample memory at the generic adapter
# default of 8.  Batch 32 materially improves Whisper throughput while staying
# well below the 140 GB capacity measured by this full198 evaluation.
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-32}"
export ASR_BATCH_SIZE

PHASE2_ITERATION="${PHASE2_ITERATION:-15381}"
PHASE3_ITERATION="${PHASE3_ITERATION:-9075}"
printf -v PHASE2_TAG 'iter_%07d' "$((10#${PHASE2_ITERATION}))"
printf -v PHASE3_TAG 'iter_%07d' "$((10#${PHASE3_ITERATION}))"
PHASE2_HF="${PHASE2_HF:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase2_unist198_${PHASE2_TAG}_hf}"
PHASE3_HF="${PHASE3_HF:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_${PHASE3_TAG}_hf}"

CONTROL_ROOT="${REPO_ROOT}/eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}"
PIPELINE_LOG="${CONTROL_ROOT}/pipeline.log"
STATUS_FILE="${CONTROL_ROOT}/status.txt"

DEV_MANIFEST="${MANIFEST_ROOT}/unist_dev_all.jsonl"
TEST_MANIFEST="${MANIFEST_ROOT}/unist_test_all.jsonl"
SMOKE_MANIFEST="${MANIFEST_ROOT}/unist_dev_smoke_3.jsonl"

P2_SMOKE="${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_dev_smoke_${RUN_ID}"
P3_SMOKE="${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_dev_smoke_${RUN_ID}"
P2_VLLM_SMOKE="${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_dev_vllm_smoke_${RUN_ID}"
P3_VLLM_SMOKE="${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_dev_vllm_smoke_${RUN_ID}"
P2_LISTEN="${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_dev_listen_${RUN_ID}"
P3_LISTEN="${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_dev_listen_${RUN_ID}"
P2_DEV="${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_dev_full_${RUN_ID}"
P3_DEV="${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_dev_full_${RUN_ID}"
P2_TEST="${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_test_full_${RUN_ID}"
P3_TEST="${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_test_full_${RUN_ID}"

mkdir -p "${CONTROL_ROOT}/logs" "${CONTROL_ROOT}/environment"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

status() {
  local message="$1"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" | tee "${STATUS_FILE}"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

gpu_compute_processes() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -u
}

wait_for_idle_gpus() {
  if [[ "${WAIT_FOR_IDLE_GPUS}" != "1" ]]; then
    return
  fi
  while [[ -n "$(gpu_compute_processes)" ]]; do
    status "waiting_for_idle_gpus pids=$(gpu_compute_processes | paste -sd, -)"
    sleep "${GPU_WAIT_INTERVAL}"
  done
}

run_hf_kind() {
  local kind="$1"
  local phase2_output="$2"
  local phase3_output="$3"
  if [[ ! -f "${phase2_output}/summary.json" || ! -f "${phase3_output}/summary.json" ]]; then
    if [[ -e "${phase2_output}" || -e "${phase3_output}" ]]; then
      echo "Incomplete ${kind} output already exists; use a new RUN_ID to preserve it." >&2
      exit 1
    fi
    status "${kind}_started"
    RUN_ID="${RUN_ID}" \
    PHASE2_ITERATION="${PHASE2_ITERATION}" \
    PHASE3_ITERATION="${PHASE3_ITERATION}" \
    PHASE2_HF="${PHASE2_HF}" \
    PHASE3_HF="${PHASE3_HF}" \
    EVAL_CUDA_VISIBLE_DEVICES="${HF_GPU:-0}" \
      "${EVAL_ROOT}/run_hf_matrix.sh" "${kind}"
    status "${kind}_complete"
  else
    status "${kind}_generation_already_complete"
  fi

  if [[ "${kind}" == "smoke" ]]; then
    status "smoke_objective_metrics_started"
    CUDA_VISIBLE_DEVICES="${HF_GPU:-0}" ENV_ROOT="${ENV_ROOT}" DEVICE="cuda:0" \
      "${EVAL_ROOT}/run_objective_metrics.sh" "${phase2_output}" \
      >"${CONTROL_ROOT}/logs/phase2_smoke_objective.log" 2>&1
    CUDA_VISIBLE_DEVICES="${HF_GPU:-0}" ENV_ROOT="${ENV_ROOT}" DEVICE="cuda:0" \
      "${EVAL_ROOT}/run_objective_metrics.sh" "${phase3_output}" \
      >"${CONTROL_ROOT}/logs/phase3_smoke_objective.log" 2>&1
    status "smoke_objective_metrics_complete"
  fi
}

run_full_one() {
  local stage="$1"
  local checkpoint="$2"
  local manifest="$3"
  local output="$4"
  local gpu="$5"
  local log="$6"
  local allow_generated_failures="$7"
  local resume=0
  if [[ -e "${output}" ]]; then
    resume=1
  fi
  EVAL_GPU_LIST="${gpu}" \
  ENV_ROOT="${ENV_ROOT}" \
  RESUME="${resume}" \
  ALLOW_GENERATED_FAILURES="${allow_generated_failures}" \
    "${EVAL_ROOT}/run_vllm_eval.sh" "${stage}" "${checkpoint}" "${manifest}" "${output}" \
    >"${log}" 2>&1
  EVAL_GPU_LIST="${gpu}" \
  ENV_ROOT="${ENV_ROOT}" \
  DEVICE="cuda:0" \
    "${EVAL_ROOT}/run_objective_metrics.sh" "${output}" \
    >>"${log}" 2>&1
}

run_pair() {
  local split="$1"
  local manifest="$2"
  local phase2_output="$3"
  local phase3_output="$4"
  local phase2_gpu="$5"
  local phase3_gpu="$6"
  local allow_generated_failures=1
  if [[ "${split}" == "vllm_smoke" ]]; then
    allow_generated_failures=0
  fi
  status "${split}_full_started"
  run_full_one phase2 "${PHASE2_HF}" "${manifest}" "${phase2_output}" "${phase2_gpu}" \
    "${CONTROL_ROOT}/logs/phase2_${split}.log" "${allow_generated_failures}" &
  local phase2_pid=$!
  run_full_one phase3 "${PHASE3_HF}" "${manifest}" "${phase3_output}" "${phase3_gpu}" \
    "${CONTROL_ROOT}/logs/phase3_${split}.log" "${allow_generated_failures}" &
  local phase3_pid=$!

  local phase2_status=0
  local phase3_status=0
  wait "${phase2_pid}" || phase2_status=$?
  wait "${phase3_pid}" || phase3_status=$?
  if [[ "${phase2_status}" -ne 0 || "${phase3_status}" -ne 0 ]]; then
    status "${split}_full_failed phase2_status=${phase2_status} phase3_status=${phase3_status}"
    return 1
  fi
  status "${split}_full_complete"
}

aggregate() {
  status "aggregate_started"
  "${ENV_ROOT}/bin/python" -m evaluation.aggregate_report \
    --run \
      "${P2_SMOKE}" "${P3_SMOKE}" \
      "${P2_VLLM_SMOKE}" "${P3_VLLM_SMOKE}" \
      "${P2_LISTEN}" "${P3_LISTEN}" \
      "${P2_DEV}" "${P3_DEV}" \
      "${P2_TEST}" "${P3_TEST}" \
    --output-dir "${CONTROL_ROOT}/report"
  cp "${MANIFEST_ROOT}/cvss_t_manifest_summary.json" "${CONTROL_ROOT}/report/cvss_t_manifest_summary.json"
  status "complete"
  touch "${CONTROL_ROOT}/COMPLETE"
}

require_dir "${PHASE2_HF}"
require_dir "${PHASE3_HF}"
require_file "${DEV_MANIFEST}"
require_file "${TEST_MANIFEST}"
require_file "${SMOKE_MANIFEST}"
require_file "${MANIFEST_ROOT}/unist_dev_listen_50.jsonl"
require_file "${MANIFEST_ROOT}/cvss_t_manifest_summary.json"

git -C "${REPO_ROOT}" rev-parse HEAD >"${CONTROL_ROOT}/environment/git_commit.txt"
git -C "${REPO_ROOT}" status --short --branch >"${CONTROL_ROOT}/environment/git_status.txt"
"${ENV_ROOT}/bin/python" -m pip freeze >"${CONTROL_ROOT}/environment/eval_pip_freeze.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
  >"${CONTROL_ROOT}/environment/gpus.csv"
sha256sum \
  "${PHASE2_HF}/model.safetensors" \
  "${PHASE3_HF}/model.safetensors" \
  "${SMOKE_MANIFEST}" \
  "${MANIFEST_ROOT}/unist_dev_listen_50.jsonl" \
  "${DEV_MANIFEST}" \
  "${TEST_MANIFEST}" \
  >"${CONTROL_ROOT}/environment/frozen_inputs.sha256"
cp "${PHASE2_HF}/export_manifest.json" "${CONTROL_ROOT}/environment/phase2_export_manifest.json"
cp "${PHASE3_HF}/export_manifest.json" "${CONTROL_ROOT}/environment/phase3_export_manifest.json"
cp "${EVAL_ROOT}/metric_models.json" "${CONTROL_ROOT}/environment/metric_models.json"

cat >"${CONTROL_ROOT}/run_paths.txt" <<EOF
RUN_ID=${RUN_ID}
PHASE2_HF=${PHASE2_HF}
PHASE3_HF=${PHASE3_HF}
P2_SMOKE=${P2_SMOKE}
P3_SMOKE=${P3_SMOKE}
P2_VLLM_SMOKE=${P2_VLLM_SMOKE}
P3_VLLM_SMOKE=${P3_VLLM_SMOKE}
P2_LISTEN=${P2_LISTEN}
P3_LISTEN=${P3_LISTEN}
P2_DEV=${P2_DEV}
P3_DEV=${P3_DEV}
P2_TEST=${P2_TEST}
P3_TEST=${P3_TEST}
REPORT=${CONTROL_ROOT}/report
ASR_BATCH_SIZE=${ASR_BATCH_SIZE}
EOF

status "initialized"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  status "preflight_complete"
  exit 0
fi
wait_for_idle_gpus
run_hf_kind smoke "${P2_SMOKE}" "${P3_SMOKE}"
run_pair vllm_smoke "${SMOKE_MANIFEST}" "${P2_VLLM_SMOKE}" "${P3_VLLM_SMOKE}" \
  "${VLLM_SMOKE_PHASE2_GPU:-0}" "${VLLM_SMOKE_PHASE3_GPU:-1}"
run_hf_kind listen "${P2_LISTEN}" "${P3_LISTEN}"
run_pair dev "${DEV_MANIFEST}" "${P2_DEV}" "${P3_DEV}" \
  "${DEV_PHASE2_GPUS:-${DEV_PHASE2_GPU:-0,1,2,3}}" \
  "${DEV_PHASE3_GPUS:-${DEV_PHASE3_GPU:-4,5,6,7}}"
run_pair test "${TEST_MANIFEST}" "${P2_TEST}" "${P3_TEST}" \
  "${TEST_PHASE2_GPUS:-${TEST_PHASE2_GPU:-0,1,2,3}}" \
  "${TEST_PHASE3_GPUS:-${TEST_PHASE3_GPU:-4,5,6,7}}"
aggregate
