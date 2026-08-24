#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${TRAIN_RUN_ID:?set the immutable epoch-2/3 RUN_ID}"

TRAIN_SESSION=${TRAIN_SESSION:-endmargin_epoch23_train}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}
TRAIN_REPORT_ROOT=${REPORT_ROOT}/extended_canaries/${TRAIN_RUN_ID}
TRAIN_SUMMARY=${TRAIN_REPORT_ROOT}/EXTENDED_CONTINUATION.json
TRAIN_LOG=${LOG_ROOT}/extended_canaries/${TRAIN_RUN_ID}.log
TRAIN_SAVE_ROOT=${CHECKPOINT_ROOT}/extended_canaries/${TRAIN_RUN_ID}
POST_LOG=${LOG_ROOT}/extended_canaries/${TRAIN_RUN_ID}.post_training.log
SELECTION=${SELECTION:-${REPORT_ROOT}/free_running_gates/free_running_gate_learning100u_20260821T142900Z/SELECTION.json}
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_learning_canary_gpu.lock

[[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${POLL_SECONDS}" -ge 10 ]] || {
  echo "POLL_SECONDS must be an integer of at least 10" >&2
  exit 2
}
[[ ! -e "${POST_LOG}" ]] || { echo "refusing to overwrite ${POST_LOG}" >&2; exit 3; }
mkdir -p "$(dirname -- "${POST_LOG}")" "$(dirname -- "${GPU_LOCK}")"
exec > >(tee -a "${POST_LOG}") 2>&1
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "waiting_for=${TRAIN_SUMMARY}"

while [[ ! -f "${TRAIN_SUMMARY}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    echo "training session ended before producing ${TRAIN_SUMMARY}" >&2
    tail -n 100 "${TRAIN_LOG}" 2>/dev/null || true
    exit 4
  fi
  latest_iteration=$(sed -n 's/.*iteration[[:space:]]*\([0-9][0-9]*\)\/[[:space:]]*2264.*/\1/p' "${TRAIN_LOG}" 2>/dev/null | tail -n 1)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) iteration=${latest_iteration:-not_started}/2264"
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" - "${TRAIN_SUMMARY}" "${TRAIN_SAVE_ROOT}" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1]).resolve()
save_root = pathlib.Path(sys.argv[2]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["schema_version"] == "uniss_e2e_extended_continuation_v1"
assert summary["status"] == "complete"
assert summary["formal_training_authorized"] is False
assert summary["additional_coverage_epochs"] == 2
assert summary["cumulative_coverage_epochs"] == 3
assert summary["train_iters"] == 2264
assert pathlib.Path(summary["epoch2_checkpoint"]).resolve() == save_root / "iter_0001207"
assert pathlib.Path(summary["epoch3_checkpoint"]).resolve() == save_root / "iter_0002264"
audit = json.loads(pathlib.Path(summary["frozen_stage_a_audit"]).read_text(encoding="utf-8"))
assert audit["status"] == "passed" and audit["exact_bitwise_match"] is True
assert all((save_root / f"iter_{iteration:07d}" / "metadata.json").is_file() for iteration in (1207, 2264))
print(f"validated_training_summary={summary_path}")
PY

exec 9>"${GPU_LOCK}"
flock 9
echo "training_gpu_lock_released_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for iteration in 1207 2264; do
  iter_tag=$(printf '%07d' "${iteration}")
  checkpoint=${TRAIN_SAVE_ROOT}/iter_${iter_tag}
  candidate_hf=${REPO_ROOT}/checkpoints/exported_hf/${EXPERIMENT_NAME}_${TRAIN_RUN_ID}_iter_${iter_tag}_hf
  gate_run_id=free_running_gate_${TRAIN_RUN_ID}_iter${iter_tag}_sem${MAX_S2S_SEMANTIC_TOKENS}
  env \
    DATA_RUN_ID="${DATA_RUN_ID}" \
    FORMAL_DATA_RUN_ID="${DATA_RUN_ID}" \
    LEARNING_RUN_ID="${TRAIN_RUN_ID}" \
    LEARNING_ITER="${iteration}" \
    MEGATRON_CHECKPOINT="${checkpoint}" \
    CANDIDATE_HF="${candidate_hf}" \
    GATE_RUN_ID="${gate_run_id}" \
    EXPORT_ATTEMPT_ID="${TRAIN_RUN_ID}_iter${iter_tag}" \
    SELECTION="${SELECTION}" \
    MAX_S2S_SEMANTIC_TOKENS="${MAX_S2S_SEMANTIC_TOKENS}" \
    POLL_SECONDS="${POLL_SECONDS}" \
    "${BASE_EXPERIMENT}/scripts/wait_for_gpu_then_export_learning_gate.sh"
done

echo "post_training_status=complete"
echo "epoch2_gate=${REPORT_ROOT}/free_running_gates/free_running_gate_${TRAIN_RUN_ID}_iter0001207_sem${MAX_S2S_SEMANTIC_TOKENS}/E2E_FREE_RUNNING_GATE.json"
echo "epoch3_gate=${REPORT_ROOT}/free_running_gates/free_running_gate_${TRAIN_RUN_ID}_iter0002264_sem${MAX_S2S_SEMANTIC_TOKENS}/E2E_FREE_RUNNING_GATE.json"
