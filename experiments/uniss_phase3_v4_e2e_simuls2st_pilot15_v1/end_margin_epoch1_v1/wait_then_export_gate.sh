#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${TRAIN_RUN_ID:?set the immutable END-margin training RUN_ID}"

TRAIN_ITERS=${TRAIN_ITERS:-1132}
POLL_SECONDS=${POLL_SECONDS:-30}
TRAIN_SESSION=${TRAIN_SESSION:-endmargin_epoch1_train}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}
TRAIN_REPORT_ROOT=${REPORT_ROOT}/extended_canaries/${TRAIN_RUN_ID}
TRAIN_SUMMARY=${TRAIN_REPORT_ROOT}/EXTENDED_CANARY.json
TRAIN_LOG=${LOG_ROOT}/extended_canaries/${TRAIN_RUN_ID}.log
TRAIN_CHECKPOINT=${CHECKPOINT_ROOT}/extended_canaries/${TRAIN_RUN_ID}/iter_$(printf '%07d' "$((10#${TRAIN_ITERS}))")
CANDIDATE_HF=${CANDIDATE_HF:-${REPO_ROOT}/checkpoints/exported_hf/${EXPERIMENT_NAME}_${TRAIN_RUN_ID}_iter_$(printf '%07d' "$((10#${TRAIN_ITERS}))")_hf}
GATE_RUN_ID=${GATE_RUN_ID:-free_running_gate_${TRAIN_RUN_ID}_sem${MAX_S2S_SEMANTIC_TOKENS}}
SELECTION=${SELECTION:-${REPORT_ROOT}/free_running_gates/free_running_gate_learning100u_20260821T142900Z/SELECTION.json}
POST_LOG=${POST_LOG:-${LOG_ROOT}/extended_canaries/${TRAIN_RUN_ID}.post_training.log}
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_learning_canary_gpu.lock

[[ "${TRAIN_ITERS}" =~ ^[0-9]+$ && "${TRAIN_ITERS}" == "1132" ]] || {
  echo "the bounded END-margin run requires exactly 1132 updates" >&2
  exit 2
}
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${POLL_SECONDS}" -ge 10 ]] || {
  echo "POLL_SECONDS must be an integer of at least 10" >&2
  exit 2
}
[[ -f "${TRAIN_LOG}" ]] || { echo "missing training log: ${TRAIN_LOG}" >&2; exit 2; }
[[ -f "${SELECTION}" ]] || { echo "missing fixed-16 selection: ${SELECTION}" >&2; exit 2; }
[[ ! -e "${POST_LOG}" ]] || { echo "refusing to overwrite ${POST_LOG}" >&2; exit 3; }

mkdir -p "$(dirname -- "${POST_LOG}")" "$(dirname -- "${GPU_LOCK}")"
exec > >(tee -a "${POST_LOG}") 2>&1
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "waiting_for=${TRAIN_SUMMARY}"

while [[ ! -f "${TRAIN_SUMMARY}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    echo "training session ended before producing ${TRAIN_SUMMARY}" >&2
    tail -n 80 "${TRAIN_LOG}" || true
    exit 4
  fi
  latest_iteration=$(sed -n 's/.*iteration[[:space:]]*\([0-9][0-9]*\)\/[[:space:]]*1132.*/\1/p' "${TRAIN_LOG}" | tail -n 1)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) iteration=${latest_iteration:-not_started}/1132"
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" - "${TRAIN_SUMMARY}" "${TRAIN_CHECKPOINT}" "${TRAIN_ITERS}" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1]).resolve()
expected_checkpoint = pathlib.Path(sys.argv[2]).resolve()
expected_iters = int(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["schema_version"] == "uniss_e2e_extended_canary_v1"
assert summary["status"] == "complete"
assert summary["coverage_epochs"] == 1
assert summary["train_iters"] == expected_iters
assert summary["formal_training_authorized"] is False
assert pathlib.Path(summary["checkpoint"]).resolve() == expected_checkpoint
assert (expected_checkpoint / "metadata.json").is_file()
audit_path = pathlib.Path(summary["frozen_stage_a_audit"]).resolve()
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert audit["schema_version"] == "uniss_e2e_frozen_stage_a_bitwise_audit_v1"
assert audit["status"] == "passed"
assert audit["exact_bitwise_match"] is True
print(f"validated_training_summary={summary_path}")
print(f"validated_checkpoint={expected_checkpoint}")
print(f"validated_frozen_audit={audit_path}")
PY

# The training shell writes its summary before releasing this lock.  Blocking
# here guarantees that export/evaluation cannot overlap its final teardown and
# prevents another in-tree E2E run from claiming the same eight GPUs.
exec 9>"${GPU_LOCK}"
flock 9
echo "training_gpu_lock_released_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

env \
  DATA_RUN_ID="${DATA_RUN_ID}" \
  FORMAL_DATA_RUN_ID="${DATA_RUN_ID}" \
  LEARNING_RUN_ID="${TRAIN_RUN_ID}" \
  LEARNING_ITER="${TRAIN_ITERS}" \
  MEGATRON_CHECKPOINT="${TRAIN_CHECKPOINT}" \
  CANDIDATE_HF="${CANDIDATE_HF}" \
  GATE_RUN_ID="${GATE_RUN_ID}" \
  EXPORT_ATTEMPT_ID="${TRAIN_RUN_ID}_iter${TRAIN_ITERS}" \
  SELECTION="${SELECTION}" \
  MAX_S2S_SEMANTIC_TOKENS="${MAX_S2S_SEMANTIC_TOKENS}" \
  POLL_SECONDS="${POLL_SECONDS}" \
  "${BASE_EXPERIMENT}/scripts/wait_for_gpu_then_export_learning_gate.sh"

echo "post_training_status=complete"
echo "candidate_hf=${CANDIDATE_HF}"
echo "gate=${REPORT_ROOT}/free_running_gates/${GATE_RUN_ID}/E2E_FREE_RUNNING_GATE.json"
