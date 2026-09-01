#!/usr/bin/env bash
# Waits for a continue-end training run to report complete, then exports the
# checkpoint to HF, runs the local-agreement free-running gate under exactly the
# stage2_paced_m1200 configuration that measured iter_0002264, computes the
# streaming metrics, builds stereo demos, and restores the GPU holder.
#
# This script only measures.  It never launches another training run: the S2
# gate decision stays with the operator, per the plan rule that a failed gate is
# recorded as a wall rather than retried with another weight setting.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
COMMIT_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_e2e_commit_policy_v1" && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

OWN_NAME=uniss_phase3_e2e_continue_end_v1

: "${TRAIN_RUN_ID:?TRAIN_RUN_ID is required}"
: "${GATE_RUN_ID:?GATE_RUN_ID is required (must be fresh)}"

TRAIN_ITER=${TRAIN_ITER:-1132}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-43200}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}
HOLDBACK=${HOLDBACK:-2}
PACE_MARGIN_MS=${PACE_MARGIN_MS:-1200}
RESTORE_GPU_HOLDER=${RESTORE_GPU_HOLDER:-1}

SPEAK_GATE=${SPEAK_GATE:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${DATA_RUN_ID}/free_running_gates/speak_decision_iter1132_la_hb2_m1200_20260831T211140Z}
BASELINE_GATE=${BASELINE_GATE:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${DATA_RUN_ID}/free_running_gates/stage2_paced_m1200_iter0002264_20260831T180448Z}
TRAIN_REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${TRAIN_RUN_ID}
TRAIN_SUMMARY=${TRAIN_REPORT_ROOT}/CONTINUE_END_RUN.json
MEGATRON_CHECKPOINT=${REPO_ROOT}/checkpoints/${OWN_NAME}/${TRAIN_RUN_ID}/iter_$(printf '%07d' "$((10#${TRAIN_ITER}))")
CANDIDATE_HF=${CANDIDATE_HF:-${REPO_ROOT}/checkpoints/exported_hf/${OWN_NAME}_${TRAIN_RUN_ID}_iter_$(printf '%07d' "$((10#${TRAIN_ITER}))")_hf}
BASE_HF=${BASE_HF:-${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf}

GATE_ROOT=${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${DATA_RUN_ID}/free_running_gates/${GATE_RUN_ID}
REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${TRAIN_RUN_ID}/gate_${GATE_RUN_ID}
OWN_LOG_ROOT=${REPO_ROOT}/logs/${OWN_NAME}
WATCH_LOG=${OWN_LOG_ROOT}/${GATE_RUN_ID}.chain.log
EXPORT_LOG=${OWN_LOG_ROOT}/${GATE_RUN_ID}.export.log

for path in "${BASE_HF}/model.safetensors" "${BASELINE_GATE}/SELECTION.json" \
            "${COMMIT_EXPERIMENT}/scripts/run_gate_local_agreement_8gpu.sh" \
            "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh"; do
  [[ -e "${path}" ]] || { echo "missing chain input: ${path}" >&2; exit 2; }
done
for path in "${GATE_ROOT}" "${REPORT_ROOT}" "${CANDIDATE_HF}" "${WATCH_LOG}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}" >&2; exit 3; }
done
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${POLL_SECONDS}" -ge 10 ]] || {
  echo "POLL_SECONDS must be an integer of at least 10" >&2; exit 2; }

mkdir -p "$(dirname -- "${WATCH_LOG}")"
exec > >(tee -a "${WATCH_LOG}") 2>&1
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "train_run=${TRAIN_RUN_ID} train_iter=${TRAIN_ITER}"
echo "waiting_for=${TRAIN_SUMMARY}"

waited=0
while [[ ! -f "${TRAIN_SUMMARY}" ]]; do
  if (( waited >= MAX_WAIT_SECONDS )); then
    echo "training did not report complete within ${MAX_WAIT_SECONDS}s" >&2
    exit 4
  fi
  sleep "${POLL_SECONDS}"
  waited=$(( waited + POLL_SECONDS ))
done
echo "training_summary_seen_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) after ${waited}s"

"${PYTHON_BIN}" - "${TRAIN_SUMMARY}" <<'PY'
import json, sys
summary = json.loads(open(sys.argv[1]).read())
if summary.get("status") != "complete":
    raise SystemExit(f"training status is {summary.get('status')!r}; refusing to gate")
print("training_status=complete")
print("final_checkpoint=" + summary["final_checkpoint"])
PY

[[ -f "${MEGATRON_CHECKPOINT}/metadata.json" ]] || {
  echo "missing megatron checkpoint: ${MEGATRON_CHECKPOINT}/metadata.json" >&2; exit 5; }

echo "waiting_for=idle_8gpu"
waited=0
while true; do
  visible=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l || true)
  if [[ "${visible}" == "8" ]]; then
    mapfile -t active_gpu_pids < <(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "[N/A]" {print $1}' | sort -u
    )
    (( ${#active_gpu_pids[@]} == 0 )) && break
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs busy: ${active_gpu_pids[*]}"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) visible_gpus=${visible}"
  fi
  if (( waited >= MAX_WAIT_SECONDS )); then
    echo "GPUs never went idle within ${MAX_WAIT_SECONDS}s" >&2; exit 6
  fi
  sleep "${POLL_SECONDS}"
  waited=$(( waited + POLL_SECONDS ))
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

echo "step=export"
"${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${BASE_HF}" \
  --megatron-path "${MEGATRON_CHECKPOINT}" \
  --hf-output "${CANDIDATE_HF}" \
  --model-type gpt 2>&1 | tee "${EXPORT_LOG}"

mkdir -p "${GATE_ROOT}" "${REPORT_ROOT}"
cp "${BASELINE_GATE}/SELECTION.json" "${GATE_ROOT}/SELECTION.json"

echo "step=fingerprint"
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "candidate_hf=${CANDIDATE_HF}" \
  --workers 12 \
  --output "${GATE_ROOT}/CANDIDATE_HF_FINGERPRINT.json" \
  | tee "${GATE_ROOT}/FINGERPRINT.stdout.json" >/dev/null

echo "step=gate holdback=${HOLDBACK} pace_margin_ms=${PACE_MARGIN_MS} sem_cap=${MAX_S2S_SEMANTIC_TOKENS}"
env \
  RUN_ID="${GATE_RUN_ID}" \
  RUN_ROOT="${GATE_ROOT}" \
  SELECTION="${GATE_ROOT}/SELECTION.json" \
  CANDIDATE_HF="${CANDIDATE_HF}" \
  CANDIDATE_FINGERPRINT="${GATE_ROOT}/CANDIDATE_HF_FINGERPRINT.json" \
  CANDIDATE_CHECKPOINT="${MEGATRON_CHECKPOINT}" \
  NUM_WORKERS=8 \
  MAX_S2S_SEMANTIC_TOKENS="${MAX_S2S_SEMANTIC_TOKENS}" \
  UNISS_E2E_MT_HOLDBACK="${HOLDBACK}" \
  UNISS_E2E_SEMANTIC_PACE=1 \
  UNISS_E2E_SEMANTIC_PACE_MARGIN_MS="${PACE_MARGIN_MS}" \
  "${COMMIT_EXPERIMENT}/scripts/run_gate_local_agreement_8gpu.sh"

echo "step=metrics"
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_e2e_commit_policy_v1.evaluation.streaming_metrics \
  --run "continue_end=${GATE_ROOT}" \
  --run "baseline_iter0002264=${BASELINE_GATE}" \
  --run "speak_decision_iter1132=${SPEAK_GATE}" \
  --offline-baseline "${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/baseline_summary.json" \
  --output "${REPORT_ROOT}/METRICS.json" \
  | tee "${REPORT_ROOT}/METRICS.stdout.txt"

echo "step=demos"
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_e2e_commit_policy_v1.evaluation.build_stereo_demos \
  --run "continue_end=${GATE_ROOT}" \
  --selection "${GATE_ROOT}/SELECTION.json" \
  --sample-id emilia_zh_0004122419 \
  --sample-id emilia_zh_0006199435 \
  --output-dir "${REPORT_ROOT}/audio" \
  --manifest "${REPORT_ROOT}/MANIFEST.json" >/dev/null

if [[ "${RESTORE_GPU_HOLDER}" == "1" ]]; then
  echo "step=restore_gpu_holder"
  bash "${HERE}/../uniss_phase3_content_first_joint_s2st_v1/scripts/start_gpu_holder.sh" || \
    echo "gpu holder restore failed; restore it manually" >&2
fi

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "gate=${GATE_ROOT}/E2E_FREE_RUNNING_GATE.json"
echo "metrics=${REPORT_ROOT}/METRICS.json"
echo "demos=${REPORT_ROOT}/MANIFEST.json"
echo "candidate_hf=${CANDIDATE_HF}"
