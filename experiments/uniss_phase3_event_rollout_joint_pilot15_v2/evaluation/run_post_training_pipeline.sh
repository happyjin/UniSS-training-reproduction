#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"

RUN_NAME="${RUN_NAME:-uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/logs/${RUN_NAME}_train_tmux.log}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/${RUN_NAME}}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-717}"
MAXIMUM_CANDIDATES="${MAXIMUM_CANDIDATES:-3}"
PROBE_VALID_SAMPLES_PER_DIRECTION="${PROBE_VALID_SAMPLES_PER_DIRECTION:-2}"
PROBE_TRAIN_SAMPLES_PER_DIRECTION="${PROBE_TRAIN_SAMPLES_PER_DIRECTION:-2}"
PROBE_RETENTION_SAMPLES_PER_DIRECTION="${PROBE_RETENTION_SAMPLES_PER_DIRECTION:-2}"
FULL_TRAIN_SAMPLES_PER_DIRECTION="${FULL_TRAIN_SAMPLES_PER_DIRECTION:-64}"
FULL_RETENTION_SAMPLES_PER_DIRECTION="${FULL_RETENTION_SAMPLES_PER_DIRECTION:-64}"
PARITY_SAMPLES_PER_DIRECTION="${PARITY_SAMPLES_PER_DIRECTION:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python}"
PIPELINE_TAG="${PIPELINE_TAG:-post_training_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/post_training_pipeline/${PIPELINE_TAG}}"

for value in \
  "${TRAIN_LOG}" \
  "${CHECKPOINT_ROOT}" \
  "${TRAIN_PYTHON}" \
  "${EVAL_DIR}/run_checkpoint_evaluation_8gpu.sh" \
  "${EVAL_DIR}/run_quality_metrics_8gpu.sh" \
  "${EVAL_DIR}/run_phase3_retention_8gpu.sh" \
  "${EVAL_DIR}/run_phase3_retention_metrics_8gpu.sh"; do
  [[ -e "${value}" ]] || { echo "Missing post-training pipeline input: ${value}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }
for value in \
  "${TRAIN_ITERATIONS}" \
  "${MAXIMUM_CANDIDATES}" \
  "${PROBE_VALID_SAMPLES_PER_DIRECTION}" \
  "${PROBE_TRAIN_SAMPLES_PER_DIRECTION}" \
  "${PROBE_RETENTION_SAMPLES_PER_DIRECTION}" \
  "${FULL_TRAIN_SAMPLES_PER_DIRECTION}" \
  "${FULL_RETENTION_SAMPLES_PER_DIRECTION}" \
  "${PARITY_SAMPLES_PER_DIRECTION}" \
  "${POLL_SECONDS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer, got ${value}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}/logs"
exec > >(tee -a "${OUTPUT_ROOT}/logs/pipeline.log") 2>&1
printf 'post-training pipeline root: %s\n' "${OUTPUT_ROOT}"

latest_logged_iteration() {
  "${TRAIN_PYTHON}" - "${TRAIN_LOG}" <<'PY'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"iteration\s+(\d+)/\s*(\d+)", text)
print(matches[-1][0] if matches else 0)
PY
}

training_process_running() {
  pgrep -f "pretrain_event_rollout_megatron.py.*--save ${CHECKPOINT_ROOT}" >/dev/null
}

while true; do
  current="$(latest_logged_iteration)"
  if grep -q '\[after training is done\]' "${TRAIN_LOG}" && \
     [[ "${current}" -ge "${TRAIN_ITERATIONS}" ]]; then
    printf 'training complete at iteration %s/%s\n' "${current}" "${TRAIN_ITERATIONS}"
    break
  fi
  if ! training_process_running; then
    echo "Training process exited before an auditable completion marker at ${current}/${TRAIN_ITERATIONS}" >&2
    exit 1
  fi
  printf 'waiting for training: iteration %s/%s\n' "${current}" "${TRAIN_ITERATIONS}"
  sleep "${POLL_SECONDS}"
done

SUMMARY_ROOT="${OUTPUT_ROOT}/validation"
mkdir -p "${SUMMARY_ROOT}"
"${TRAIN_PYTHON}" \
  -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.summarize_validation \
  --log "${TRAIN_LOG}" --checkpoint-root "${CHECKPOINT_ROOT}" \
  --json "${SUMMARY_ROOT}/validation_checkpoints.json" \
  --markdown "${SUMMARY_ROOT}/validation_checkpoints.md"
"${TRAIN_PYTHON}" \
  -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.shortlist_checkpoints \
  --validation-summary "${SUMMARY_ROOT}/validation_checkpoints.json" \
  --maximum-candidates "${MAXIMUM_CANDIDATES}" \
  --json "${SUMMARY_ROOT}/checkpoint_shortlist.json" \
  --markdown "${SUMMARY_ROOT}/checkpoint_shortlist.md"

mapfile -t CANDIDATES < <(
  "${TRAIN_PYTHON}" - "${SUMMARY_ROOT}/checkpoint_shortlist.json" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in report["candidates"]:
    print(int(row["iteration"]))
PY
)
[[ "${#CANDIDATES[@]}" -gt 0 ]] || { echo "Shortlist produced no candidates" >&2; exit 1; }
printf 'probe candidates: %s\n' "${CANDIDATES[*]}"

probe_selector_args=()
for iteration in "${CANDIDATES[@]}"; do
  formatted="$(printf '%07d' "${iteration}")"
  candidate_root="${OUTPUT_ROOT}/probe/iter_${formatted}"
  runtime_root="${candidate_root}/checkpoint_evaluation"
  retention_root="${candidate_root}/phase3_retention"
  mkdir -p "${candidate_root}"

  RUN_NAME="${RUN_NAME}" ITERATION="${iteration}" GPU_LIST="${GPU_LIST}" \
    OUTPUT_ROOT="${runtime_root}" EVAL_TAG="${PIPELINE_TAG}_probe_iter_${formatted}" \
    VALID_SAMPLES_PER_DIRECTION="${PROBE_VALID_SAMPLES_PER_DIRECTION}" \
    TRAIN_SAMPLES_PER_DIRECTION="${PROBE_TRAIN_SAMPLES_PER_DIRECTION}" \
    PARITY_SAMPLES_PER_DIRECTION="${PARITY_SAMPLES_PER_DIRECTION}" \
    "${EVAL_DIR}/run_checkpoint_evaluation_8gpu.sh"
  GPU_LIST="${GPU_LIST}" "${EVAL_DIR}/run_quality_metrics_8gpu.sh" "${runtime_root}"

  RUN_NAME="${RUN_NAME}" ITERATION="${iteration}" GPU_LIST="${GPU_LIST}" \
    OUTPUT_ROOT="${retention_root}" TAG="${PIPELINE_TAG}_probe_iter_${formatted}" \
    SAMPLES_PER_DIRECTION="${PROBE_RETENTION_SAMPLES_PER_DIRECTION}" \
    "${EVAL_DIR}/run_phase3_retention_8gpu.sh"
  GPU_LIST="${GPU_LIST}" \
    "${EVAL_DIR}/run_phase3_retention_metrics_8gpu.sh" "${retention_root}"

  probe_selector_args+=(--candidate "${iteration}" "${runtime_root}" "${retention_root}")
done

"${TRAIN_PYTHON}" \
  -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.select_final_checkpoint \
  "${probe_selector_args[@]}" --output-root "${OUTPUT_ROOT}/probe_selection"

SELECTED_ITERATION="$(
  "${TRAIN_PYTHON}" - "${OUTPUT_ROOT}/probe_selection/checkpoint_selection.json" <<'PY'
import json, sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["selected_iteration"]
print("" if value is None else int(value))
PY
)"
if [[ -z "${SELECTED_ITERATION}" ]]; then
  echo "No probe checkpoint passed all hard gates; full evaluation was not started."
  exit 0
fi

formatted="$(printf '%07d' "${SELECTED_ITERATION}")"
FINAL_ROOT="${OUTPUT_ROOT}/final/iter_${formatted}"
FINAL_RUNTIME_ROOT="${FINAL_ROOT}/checkpoint_evaluation"
FINAL_RETENTION_ROOT="${FINAL_ROOT}/phase3_retention"
mkdir -p "${FINAL_ROOT}"

RUN_NAME="${RUN_NAME}" ITERATION="${SELECTED_ITERATION}" GPU_LIST="${GPU_LIST}" \
  OUTPUT_ROOT="${FINAL_RUNTIME_ROOT}" EVAL_TAG="${PIPELINE_TAG}_final_iter_${formatted}" \
  VALID_SAMPLES_PER_DIRECTION= \
  TRAIN_SAMPLES_PER_DIRECTION="${FULL_TRAIN_SAMPLES_PER_DIRECTION}" \
  PARITY_SAMPLES_PER_DIRECTION="${PARITY_SAMPLES_PER_DIRECTION}" \
  "${EVAL_DIR}/run_checkpoint_evaluation_8gpu.sh"
GPU_LIST="${GPU_LIST}" \
  "${EVAL_DIR}/run_quality_metrics_8gpu.sh" "${FINAL_RUNTIME_ROOT}"

RUN_NAME="${RUN_NAME}" ITERATION="${SELECTED_ITERATION}" GPU_LIST="${GPU_LIST}" \
  OUTPUT_ROOT="${FINAL_RETENTION_ROOT}" TAG="${PIPELINE_TAG}_final_iter_${formatted}" \
  SAMPLES_PER_DIRECTION="${FULL_RETENTION_SAMPLES_PER_DIRECTION}" \
  "${EVAL_DIR}/run_phase3_retention_8gpu.sh"
GPU_LIST="${GPU_LIST}" \
  "${EVAL_DIR}/run_phase3_retention_metrics_8gpu.sh" "${FINAL_RETENTION_ROOT}"

"${TRAIN_PYTHON}" \
  -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.select_final_checkpoint \
  --candidate "${SELECTED_ITERATION}" "${FINAL_RUNTIME_ROOT}" "${FINAL_RETENTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}/final_selection"

printf 'post-training evaluation complete: %s\n' "${OUTPUT_ROOT}"
