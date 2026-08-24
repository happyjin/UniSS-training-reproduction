#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"

PARENT_RUN_ID=${PARENT_RUN_ID:-endmargin_epoch1_15shard_20260824T111053Z}
TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID:-task_pool_formal_p4_20260820T154500Z}
TEACHER_RUN_ID=${TEACHER_RUN_ID:-teacher_cache_formal_p4_20260820T154500Z}
STRUCTURAL_CANARY_RUN_ID=${STRUCTURAL_CANARY_RUN_ID:-post_task_pool_canary_p4_replayfix_w0_20260821T085848Z}
MASTER_PORT=${MASTER_PORT:-29925}
RUN_SEED=${RUN_SEED:-20260819}
[[ "${RUN_SEED}" == "20260819" ]] || {
  echo "epoch-2/3 continuation requires audited shuffle seed 20260819" >&2
  exit 2
}

TRAIN_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_train/BUILD_COMPLETE.json
VALID_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_valid/BUILD_COMPLETE.json
V1_TRAIN_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_train/AUDIT.json
PHASE3_TRAIN_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_train/AUDIT.json
V1_VALID_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_valid/AUDIT.json
PHASE3_VALID_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_valid/AUDIT.json
CANARY_REPORT=${REPORT_ROOT}/post_task_pool_canaries/${STRUCTURAL_CANARY_RUN_ID}/CANARY_REPORT.json

PARENT_REPORT_ROOT=${REPORT_ROOT}/extended_canaries/${PARENT_RUN_ID}
PARENT_SUMMARY=${PARENT_REPORT_ROOT}/EXTENDED_CANARY.json
PARENT_GATE=${REPORT_ROOT}/free_running_gates/free_running_gate_${PARENT_RUN_ID}_sem384/E2E_FREE_RUNNING_GATE.json
PARENT_SAVE_ROOT=${CHECKPOINT_ROOT}/extended_canaries/${PARENT_RUN_ID}
PARENT_CHECKPOINT=${PARENT_SAVE_ROOT}/iter_0001132

RUN_REPORT_ROOT=${REPORT_ROOT}/extended_canaries/${RUN_ID}
RUN_LOG=${LOG_ROOT}/extended_canaries/${RUN_ID}.log
RUN_SAVE_DIR=${CHECKPOINT_ROOT}/extended_canaries/${RUN_ID}
RUN_TENSORBOARD_DIR=${TENSORBOARD_ROOT}/extended_canaries/${RUN_ID}
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json
RUN_FINGERPRINTS=${RUN_REPORT_ROOT}/CONTINUATION_CHECKPOINT_FINGERPRINT.json
CONTINUATION_AUDIT=${RUN_REPORT_ROOT}/CONTINUATION_INPUT_AUDIT.json
FROZEN_AUDIT=${RUN_REPORT_ROOT}/FROZEN_STAGE_A_BITWISE_AUDIT.json
RUN_SUMMARY=${RUN_REPORT_ROOT}/EXTENDED_CONTINUATION.json
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_learning_canary_gpu.lock

required=(
  "${TRAIN_REPORT}"
  "${VALID_REPORT}"
  "${V1_TRAIN_AUDIT}"
  "${PHASE3_TRAIN_AUDIT}"
  "${V1_VALID_AUDIT}"
  "${PHASE3_VALID_AUDIT}"
  "${CANARY_REPORT}"
  "${PARENT_SUMMARY}"
  "${PARENT_GATE}"
  "${PARENT_CHECKPOINT}/metadata.json"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing epoch-2/3 input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || {
    echo "refusing to overwrite epoch-2/3 output: ${path}" >&2
    exit 4
  }
done

mkdir -p "$(dirname -- "${GPU_LOCK}")" "${RUN_REPORT_ROOT}"
exec 9>"${GPU_LOCK}"
flock -n 9 || { echo "another E2E research run owns the GPU lock" >&2; exit 5; }
mapfile -t active_gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 != "[N/A]" {print $1}' | sort -u
)
if (( ${#active_gpu_pids[@]} > 0 )); then
  printf 'GPUs are busy; refusing to interfere with PIDs: %s\n' "${active_gpu_pids[*]}" >&2
  exit 6
fi

"${PYTHON_BIN}" - "${PARENT_SUMMARY}" "${PARENT_GATE}" "${PARENT_CHECKPOINT}" "${CONTINUATION_AUDIT}" <<'PY'
import json
import pathlib
import sys

summary_path, gate_path, checkpoint_path, output_path = map(pathlib.Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
checkpoint = checkpoint_path.resolve()
assert summary["schema_version"] == "uniss_e2e_extended_canary_v1"
assert summary["status"] == "complete"
assert summary["coverage_epochs"] == 1 and summary["train_iters"] == 1132
assert summary["formal_training_authorized"] is False
assert pathlib.Path(summary["checkpoint"]).resolve() == checkpoint
assert gate["schema_version"] == "uniss_phase3_v4_e2e_free_running_gate_v1"
assert gate["formal_training_authorized"] is False
metrics = gate["metrics"]
checks = gate["checks"]
assert checks["e_asr_cmn_retained"] and checks["e_asr_eng_retained"]
assert metrics["e_asr"]["cmn"]["error_rate"] < 0.20657276995305165
assert metrics["e_asr"]["eng"]["error_rate"] < 0.47096774193548385
assert metrics["e_mt"]["gold_source"]["target_coverage_mean"] > 0.144563795956459
assert metrics["e_mt"]["free_running_source"]["target_coverage_mean"] > 0.10769597627613932
assert metrics["e_s2s_free"]["malformed_segments"] < 27
value = {
    "schema_version": "uniss_e2e_endmargin_continuation_input_audit_v1",
    "status": "passed",
    "formal_training_authorized": False,
    "parent_summary": str(summary_path.resolve()),
    "parent_gate": str(gate_path.resolve()),
    "parent_checkpoint": str(checkpoint),
    "evidence": {
        "cmn_asr_error_rate": metrics["e_asr"]["cmn"]["error_rate"],
        "eng_asr_error_rate": metrics["e_asr"]["eng"]["error_rate"],
        "gold_mt_coverage_mean": metrics["e_mt"]["gold_source"]["target_coverage_mean"],
        "free_mt_coverage_mean": metrics["e_mt"]["free_running_source"]["target_coverage_mean"],
        "malformed_s2s_segments": metrics["e_s2s_free"]["malformed_segments"],
        "non_silent_s2s_samples": metrics["e_s2s_free"]["non_silent_pcm"],
    },
}
output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1=${PARENT_CHECKPOINT}" \
  --workers 12 \
  --output "${RUN_FINGERPRINTS}" >/dev/null

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.compute_geometry \
  --task-pool-report "${TRAIN_REPORT}" \
  --global-batch-size 128 \
  --coverage-epochs 2 \
  --seed "${RUN_SEED}" \
  --output "${RUN_GEOMETRY}" >/dev/null

read -r train_iters warmup_iters epoch2_iter < <(
  "${PYTHON_BIN}" - "${RUN_GEOMETRY}" <<'PY'
import json
import sys
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.schedule import family_blocks
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import FAMILY_INTERLEAVED
value = json.load(open(sys.argv[1], encoding="utf-8"))
blocks = family_blocks(value["train_iters"], seed=value["seed"])
required = (value["family_records"][FAMILY_INTERLEAVED] + value["global_batch_size"] - 1) // value["global_batch_size"]
positions = [index + 1 for index, family in enumerate(blocks) if family == FAMILY_INTERLEAVED]
print(value["train_iters"], value["warmup_updates"], positions[required - 1])
PY
)
[[ "${train_iters}" == "2264" && "${warmup_iters}" == "68" && "${epoch2_iter}" == "1207" ]] || {
  echo "unexpected two-coverage geometry: ${train_iters}/${warmup_iters}/${epoch2_iter}" >&2
  exit 7
}

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
env \
  DATA_RUN_ID="${DATA_RUN_ID}" \
  RUN_ID="${RUN_ID}" \
  RUN_TRAIN_BUILD_REPORT="${TRAIN_REPORT}" \
  RUN_VALID_BUILD_REPORT="${VALID_REPORT}" \
  RUN_V1_TRAIN_CACHE_AUDIT="${V1_TRAIN_AUDIT}" \
  RUN_PHASE3_TRAIN_CACHE_AUDIT="${PHASE3_TRAIN_AUDIT}" \
  RUN_V1_VALID_CACHE_AUDIT="${V1_VALID_AUDIT}" \
  RUN_PHASE3_VALID_CACHE_AUDIT="${PHASE3_VALID_AUDIT}" \
  RUN_SAVE_DIR="${RUN_SAVE_DIR}" \
  RUN_TENSORBOARD_DIR="${RUN_TENSORBOARD_DIR}" \
  RUN_LOG="${RUN_LOG}" \
  RUN_GEOMETRY="${RUN_GEOMETRY}" \
  RUN_FINGERPRINTS="${RUN_FINGERPRINTS}" \
  RUN_LOAD="${PARENT_SAVE_ROOT}" \
  RUN_NPROC=8 \
  RUN_MBS=2 \
  RUN_GBS=128 \
  RUN_COVERAGE_EPOCHS=2 \
  RUN_NUM_WORKERS=0 \
  RUN_MASTER_PORT="${MASTER_PORT}" \
  RUN_SAVE_INTERVAL="${epoch2_iter}" \
  RUN_EVAL_ITERS=0 \
  RUN_EVAL_INTERVAL="${epoch2_iter}" \
  RUN_LOG_INTERVAL=1 \
  RUN_SEED="${RUN_SEED}" \
  RUN_EXTENDED_CANARY=1 \
  RUN_CANARY_REPORT="${CANARY_REPORT}" \
  RUN_TRAIN_ITERS="${train_iters}" \
  RUN_WARMUP_ITERS="${warmup_iters}" \
  RUN_AUDIT_GRADIENTS=1 \
  RUN_CONTENT_END_WEIGHT=0.0 \
  RUN_SEMANTIC_END_WEIGHT=0.5 \
  RUN_SEMANTIC_END_MARGIN_WEIGHT=0.25 \
  RUN_SEMANTIC_END_LOGIT_MARGIN=2.0 \
  RUN_SEMANTIC_ROLLIN_END_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_END_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_CONTINUE_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_CONTINUE_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_BOUNDARY_BINARY_WEIGHT=0.0 \
  RUN_SEMANTIC_PREFIX_CORRUPTION_RATE=0.0 \
  RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE=0.0 \
  RUN_VERIFY_DATASET_SHA256=0 \
  RUN_VERIFY_CACHE_SHA256=0 \
  "${BASE_EXPERIMENT}/scripts/run_e2e_megatron.sh"

EPOCH2_CHECKPOINT=${RUN_SAVE_DIR}/iter_$(printf '%07d' "${epoch2_iter}")
EPOCH3_CHECKPOINT=${RUN_SAVE_DIR}/iter_0002264
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.audit_frozen_stage_a \
  --reference "${V1_CHECKPOINT}" \
  --candidate "epoch2=${EPOCH2_CHECKPOINT}" \
  --candidate "epoch3=${EPOCH3_CHECKPOINT}" \
  --output "${FROZEN_AUDIT}" >/dev/null
ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

jq -n \
  --arg started_at "${started_at}" \
  --arg ended_at "${ended_at}" \
  --arg run_id "${RUN_ID}" \
  --arg parent_checkpoint "${PARENT_CHECKPOINT}" \
  --arg epoch2_checkpoint "${EPOCH2_CHECKPOINT}" \
  --arg epoch3_checkpoint "${EPOCH3_CHECKPOINT}" \
  --arg tensorboard "${RUN_TENSORBOARD_DIR}" \
  --arg log "${RUN_LOG}" \
  --arg gpu_csv "${RUN_LOG%.log}.gpu.csv" \
  --arg frozen_audit "${FROZEN_AUDIT}" \
  --arg continuation_audit "${CONTINUATION_AUDIT}" \
  --arg structural_canary "${CANARY_REPORT}" \
  --argjson train_iters "${train_iters}" \
  '{schema_version:"uniss_e2e_extended_continuation_v1",status:"complete",formal_training_authorized:false,started_at:$started_at,ended_at:$ended_at,run_id:$run_id,parent_coverage_epochs:1,additional_coverage_epochs:2,cumulative_coverage_epochs:3,train_iters:$train_iters,parent_checkpoint:$parent_checkpoint,epoch2_checkpoint:$epoch2_checkpoint,epoch3_checkpoint:$epoch3_checkpoint,tensorboard:$tensorboard,log:$log,gpu_csv:$gpu_csv,frozen_stage_a_audit:$frozen_audit,continuation_input_audit:$continuation_audit,structural_canary:$structural_canary,next_required_gate:"fixed_16_epoch2_and_epoch3_free_running_validation"}' \
  > "${RUN_SUMMARY}"

echo "extended_continuation_status=complete"
echo "epoch2_checkpoint=${EPOCH2_CHECKPOINT}"
echo "epoch3_checkpoint=${EPOCH3_CHECKPOINT}"
echo "report=${RUN_SUMMARY}"
echo "tensorboard=${RUN_TENSORBOARD_DIR}"
