#!/usr/bin/env bash
# One additional coverage epoch from endmargin_epoch23 iter_0002264 with the
# roll-in END/CONTINUE supervision opened.  Only RUN_* variables differ from the
# established epoch-2/3 continuation; the base experiment's own Megatron
# launcher does the work and nothing in it is modified.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"

PARENT_RUN_ID=${PARENT_RUN_ID:-endmargin_epoch23_15shard_20260824T190227Z}
PARENT_ITER=${PARENT_ITER:-0002264}
TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID:-task_pool_formal_p4_20260820T154500Z}
TEACHER_RUN_ID=${TEACHER_RUN_ID:-teacher_cache_formal_p4_20260820T154500Z}
STRUCTURAL_CANARY_RUN_ID=${STRUCTURAL_CANARY_RUN_ID:-post_task_pool_canary_p4_replayfix_w0_20260821T085848Z}
MASTER_PORT=${MASTER_PORT:-29937}
RUN_SEED=${RUN_SEED:-20260819}
COVERAGE_EPOCHS=${COVERAGE_EPOCHS:-1}

# --- objective: unchanged teacher-forced END, newly opened roll-in decision ---
# semantic_boundary_binary is an ALTERNATIVE to the margin family, not an
# addition: pretrain_e2e_megatron.py:546-566 refuses it whenever any of
# semantic_end_ce / semantic_end_margin / semantic_rollin_end_ce /
# semantic_rollin_end_margin / semantic_rollin_continue_decision_margin /
# semantic_rollin_continue_margin / semantic_continue_margin is non-zero,
# because both supervise the same END-versus-CONTINUE decision.  This run keeps
# the parent's teacher-forced END terms so the continuation stays comparable and
# is a strict superset of the parent objective, so the binary term stays off.
# Mutually exclusive with boundary roll-in; the trainer refuses both.

# --- what this experiment changes -------------------------------------------
# One weight, and the retreat of a family that three runs have falsified.
#
# boundary_eos goes from 0.10 to 1.0.  It has never been changed: every script
# in this repository leaves it at the CLI default.  In the interleaved family
# the boundary bucket -- WAIT_READ, WRITE_GENERATE, the TASK_* family choice,
# END_CONTENT, END_SEMANTIC, language and speed -- is 32.8% of supervised tokens
# and receives 4.7% of the gradient, while semantic tokens are 60.2% of tokens
# and take 85.8% of it.  Under uniform weighting those become 32.8% and 60.2%.
# UniSS's own paper reports pure next-token cross-entropy with no auxiliary
# losses for the model this project is built on, so this is that recipe applied
# to the decision tokens.
#
# Every margin and roll-in term goes to zero.  Three runs have now moved the
# inference-time speak decision monotonically the wrong way, -2.88 to -3.75 to
# -4.97, while their gold-row separation moved correctly, so the family is
# retired rather than retuned.
#
# The two distillation KLs, replay_ce and commit_consistency are KEPT at their
# established values.  Both KLs measurably fall during training -- phase3_kl by
# 0.181 and v1_asr_kl by 0.013 over the last run -- so they are working
# anti-forgetting and this run changes one thing, not ten.
BOUNDARY_EOS_WEIGHT=${BOUNDARY_EOS_WEIGHT:-1.0}
SEMANTIC_END_WEIGHT=${SEMANTIC_END_WEIGHT:-0.0}
SEMANTIC_END_MARGIN_WEIGHT=${SEMANTIC_END_MARGIN_WEIGHT:-0.0}
SEMANTIC_END_LOGIT_MARGIN=${SEMANTIC_END_LOGIT_MARGIN:-0.0}
ROLLIN_END_WEIGHT=${ROLLIN_END_WEIGHT:-0.0}
ROLLIN_END_MARGIN_WEIGHT=${ROLLIN_END_MARGIN_WEIGHT:-0.0}
ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT=${ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT:-0.0}
ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN=${ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN:-0.0}
ROLLIN_CONTINUE_MARGIN_WEIGHT=${ROLLIN_CONTINUE_MARGIN_WEIGHT:-0.0}
ROLLIN_CONTINUE_LOGIT_MARGIN=${ROLLIN_CONTINUE_LOGIT_MARGIN:-0.0}
ROLLIN_CONTINUE_TAIL=${ROLLIN_CONTINUE_TAIL:-12}
ROLLIN_CONTINUE_RATIO=${ROLLIN_CONTINUE_RATIO:-0.5}
CONTINUE_MARGIN_WEIGHT=${CONTINUE_MARGIN_WEIGHT:-0.0}
CONTINUE_LOGIT_MARGIN=${CONTINUE_LOGIT_MARGIN:-0.0}
CONTINUE_TAIL=${CONTINUE_TAIL:-12}
BOUNDARY_BINARY_WEIGHT=${BOUNDARY_BINARY_WEIGHT:-0.0}
BOUNDARY_BINARY_LOGIT_MARGIN=${BOUNDARY_BINARY_LOGIT_MARGIN:-0.0}
BOUNDARY_ROLLIN_RATE=${BOUNDARY_ROLLIN_RATE:-0.0}
BOUNDARY_ROLLIN_RAMP_UPDATES=${BOUNDARY_ROLLIN_RAMP_UPDATES:-0}
PREFIX_CORRUPTION_RATE=${PREFIX_CORRUPTION_RATE:-0.0}

# A roll-in weight with a zero rate is a silently dead loss: the masks it
# selects on are empty, exactly as real_prefix_kd / prefix_stability /
# speaker_consistency were for all 717 updates of the content-first run.
roll_in_enabled=$("${PYTHON_BIN}" - "${ROLLIN_END_WEIGHT}" "${ROLLIN_END_MARGIN_WEIGHT}" \
  "${ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT}" "${ROLLIN_CONTINUE_MARGIN_WEIGHT}" <<'PY'
import sys
print(int(any(float(value) > 0.0 for value in sys.argv[1:])))
PY
)
rate_positive=$("${PYTHON_BIN}" -c "import sys; print(int(float(sys.argv[1]) > 0.0))" "${BOUNDARY_ROLLIN_RATE}")
if [[ "${roll_in_enabled}" == "1" && "${rate_positive}" != "1" ]]; then
  echo "a roll-in weight is non-zero but BOUNDARY_ROLLIN_RATE is 0: the loss would be identically zero" >&2
  exit 2
fi

# Fail here rather than eight ranks deep in Megatron: the trainer enforces the
# same rule at pretrain_e2e_megatron.py:546-566.
binary_positive=$("${PYTHON_BIN}" -c "import sys; print(int(float(sys.argv[1]) > 0.0))" "${BOUNDARY_BINARY_WEIGHT}")
margin_family_positive=$("${PYTHON_BIN}" -c "import sys; print(int(any(float(v) != 0.0 for v in sys.argv[1:])))" \
  "${SEMANTIC_END_WEIGHT}" "${SEMANTIC_END_MARGIN_WEIGHT}" "${ROLLIN_END_WEIGHT}" \
  "${ROLLIN_END_MARGIN_WEIGHT}" "${ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT}" \
  "${ROLLIN_CONTINUE_MARGIN_WEIGHT}")
if [[ "${binary_positive}" == "1" && "${margin_family_positive}" == "1" ]]; then
  echo "semantic_boundary_binary replaces the END/CONTINUE margin family; set those weights to 0" >&2
  exit 2
fi

TRAIN_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_train/BUILD_COMPLETE.json
VALID_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_valid/BUILD_COMPLETE.json
V1_TRAIN_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_train/AUDIT.json
PHASE3_TRAIN_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_train/AUDIT.json
V1_VALID_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_valid/AUDIT.json
PHASE3_VALID_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_valid/AUDIT.json
CANARY_REPORT=${REPORT_ROOT}/post_task_pool_canaries/${STRUCTURAL_CANARY_RUN_ID}/CANARY_REPORT.json

PARENT_SUMMARY=${REPORT_ROOT}/extended_canaries/${PARENT_RUN_ID}/EXTENDED_CONTINUATION.json
PARENT_GATE=${REPORT_ROOT}/free_running_gates/free_running_gate_${PARENT_RUN_ID}_iter${PARENT_ITER}_sem384/E2E_FREE_RUNNING_GATE.json
PARENT_SAVE_ROOT=${CHECKPOINT_ROOT}/extended_canaries/${PARENT_RUN_ID}
PARENT_CHECKPOINT=${PARENT_SAVE_ROOT}/iter_${PARENT_ITER}

OWN_NAME=uniss_phase3_e2e_uniform_ce_v1
RUN_REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${RUN_ID}
RUN_LOG=${REPO_ROOT}/logs/${OWN_NAME}/${RUN_ID}.log
RUN_SAVE_DIR=${REPO_ROOT}/checkpoints/${OWN_NAME}/${RUN_ID}
RUN_TENSORBOARD_DIR=${REPO_ROOT}/runs/${OWN_NAME}/tensorboard/${RUN_ID}
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json
RUN_FINGERPRINTS=${RUN_REPORT_ROOT}/PARENT_CHECKPOINT_FINGERPRINT.json
CONTINUATION_AUDIT=${RUN_REPORT_ROOT}/CONTINUATION_INPUT_AUDIT.json
FROZEN_AUDIT=${RUN_REPORT_ROOT}/FROZEN_STAGE_A_BITWISE_AUDIT.json
RUN_SUMMARY=${RUN_REPORT_ROOT}/UNIFORM_CE_RUN.json
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_learning_canary_gpu.lock

required=(
  "${TRAIN_REPORT}" "${VALID_REPORT}"
  "${V1_TRAIN_AUDIT}" "${PHASE3_TRAIN_AUDIT}" "${V1_VALID_AUDIT}" "${PHASE3_VALID_AUDIT}"
  "${CANARY_REPORT}" "${PARENT_SUMMARY}" "${PARENT_GATE}"
  "${PARENT_CHECKPOINT}/metadata.json"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing continuation input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite output: ${path}" >&2; exit 4; }
done

mkdir -p "$(dirname -- "${GPU_LOCK}")" "${RUN_REPORT_ROOT}" "$(dirname -- "${RUN_LOG}")"
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
assert summary["schema_version"] == "uniss_e2e_extended_continuation_v1"
assert summary["status"] == "complete"
assert summary["cumulative_coverage_epochs"] == 3 and summary["train_iters"] == 2264
assert summary["formal_training_authorized"] is False
assert pathlib.Path(summary["epoch3_checkpoint"]).resolve() == checkpoint
assert gate["schema_version"] == "uniss_phase3_v4_e2e_free_running_gate_v1"
assert gate["formal_training_authorized"] is False
metrics, checks = gate["metrics"], gate["checks"]
# The parent must not be worse than the epoch-1 numbers this lineage justified
# its own continuation with.
assert checks["e_asr_cmn_retained"] and checks["e_asr_eng_retained"]
assert metrics["e_asr"]["cmn"]["error_rate"] <= 0.11737089201877934
assert metrics["e_asr"]["eng"]["error_rate"] <= 0.3096774193548387
assert metrics["e_mt"]["gold_source"]["target_coverage_mean"] >= 0.1979623071674702
assert metrics["e_s2s_free"]["malformed_segments"] <= 10
output_path.write_text(
    json.dumps(
        {
            "schema_version": "uniss_e2e_uniform_ce_input_audit_v1",
            "status": "passed",
            "formal_training_authorized": False,
            "parent_summary": str(summary_path.resolve()),
            "parent_gate": str(gate_path.resolve()),
            "parent_checkpoint": str(checkpoint),
            "evidence": {
                "cmn_asr_error_rate": metrics["e_asr"]["cmn"]["error_rate"],
                "eng_asr_error_rate": metrics["e_asr"]["eng"]["error_rate"],
                "gold_mt_coverage_mean": metrics["e_mt"]["gold_source"][
                    "target_coverage_mean"
                ],
                "malformed_s2s_segments": metrics["e_s2s_free"]["malformed_segments"],
                "natural_eos_note": "0.50 at iters 1132, 1207 and 2264 alike",
            },
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1=${PARENT_CHECKPOINT}" --workers 12 --output "${RUN_FINGERPRINTS}" >/dev/null

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.compute_geometry \
  --task-pool-report "${TRAIN_REPORT}" --global-batch-size 128 \
  --coverage-epochs "${COVERAGE_EPOCHS}" --seed "${RUN_SEED}" \
  --output "${RUN_GEOMETRY}" >/dev/null

read -r train_iters warmup_iters < <(
  "${PYTHON_BIN}" - "${RUN_GEOMETRY}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["train_iters"], value["warmup_updates"])
PY
)
echo "geometry: train_iters=${train_iters} warmup=${warmup_iters} coverage_epochs=${COVERAGE_EPOCHS}"

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
  RUN_COVERAGE_EPOCHS="${COVERAGE_EPOCHS}" \
  RUN_NUM_WORKERS="${NUM_WORKERS:-2}" \
  RUN_MASTER_PORT="${MASTER_PORT}" \
  RUN_SAVE_INTERVAL="${SAVE_INTERVAL:-200}" \
  RUN_EVAL_ITERS=0 \
  RUN_EVAL_INTERVAL="${train_iters}" \
  RUN_LOG_INTERVAL=1 \
  RUN_SEED="${RUN_SEED}" \
  RUN_EXTENDED_CANARY=1 \
  RUN_CANARY_REPORT="${CANARY_REPORT}" \
  RUN_TRAIN_ITERS="${train_iters}" \
  RUN_WARMUP_ITERS="${warmup_iters}" \
  RUN_AUDIT_GRADIENTS=1 \
  RUN_CONTENT_END_WEIGHT=0.0 \
  RUN_SEMANTIC_END_WEIGHT="${SEMANTIC_END_WEIGHT}" \
  RUN_SEMANTIC_END_MARGIN_WEIGHT="${SEMANTIC_END_MARGIN_WEIGHT}" \
  RUN_SEMANTIC_END_LOGIT_MARGIN="${SEMANTIC_END_LOGIT_MARGIN}" \
  RUN_SEMANTIC_ROLLIN_END_WEIGHT="${ROLLIN_END_WEIGHT}" \
  RUN_SEMANTIC_ROLLIN_END_MARGIN_WEIGHT="${ROLLIN_END_MARGIN_WEIGHT}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT="${ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN="${ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_MARGIN_WEIGHT="${ROLLIN_CONTINUE_MARGIN_WEIGHT}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_LOGIT_MARGIN="${ROLLIN_CONTINUE_LOGIT_MARGIN}" \
  RUN_BOUNDARY_EOS_WEIGHT="${BOUNDARY_EOS_WEIGHT}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_TAIL="${ROLLIN_CONTINUE_TAIL}" \
  RUN_SEMANTIC_ROLLIN_CONTINUE_RATIO="${ROLLIN_CONTINUE_RATIO}" \
  RUN_SEMANTIC_CONTINUE_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_BOUNDARY_BINARY_WEIGHT="${BOUNDARY_BINARY_WEIGHT}" \
  RUN_SEMANTIC_BOUNDARY_BINARY_LOGIT_MARGIN="${BOUNDARY_BINARY_LOGIT_MARGIN}" \
  RUN_SEMANTIC_PREFIX_CORRUPTION_RATE="${PREFIX_CORRUPTION_RATE}" \
  RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE="${BOUNDARY_ROLLIN_RATE}" \
  RUN_SEMANTIC_BOUNDARY_ROLLIN_RAMP_UPDATES="${BOUNDARY_ROLLIN_RAMP_UPDATES}" \
  RUN_VERIFY_DATASET_SHA256=0 \
  RUN_VERIFY_CACHE_SHA256=0 \
  "${HERE}/scripts/run_e2e_megatron_uniform.sh"

FINAL_CHECKPOINT=${RUN_SAVE_DIR}/iter_$(printf '%07d' "${train_iters}")
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.audit_frozen_stage_a \
  --reference "${V1_CHECKPOINT}" \
  --candidate "uniform_ce=${FINAL_CHECKPOINT}" \
  --output "${FROZEN_AUDIT}" >/dev/null
ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"${PYTHON_BIN}" - <<PY
import json, pathlib
pathlib.Path("${RUN_SUMMARY}").write_text(json.dumps({
  "schema_version": "uniss_e2e_uniform_ce_v1",
  "status": "complete",
  "formal_training_authorized": False,
  "started_at": "${started_at}",
  "ended_at": "${ended_at}",
  "run_id": "${RUN_ID}",
  "parent_checkpoint": "${PARENT_CHECKPOINT}",
  "final_checkpoint": "${FINAL_CHECKPOINT}",
  "additional_coverage_epochs": ${COVERAGE_EPOCHS},
  "cumulative_coverage_epochs": 3 + ${COVERAGE_EPOCHS},
  "train_iters": ${train_iters},
  "objective": {
    "semantic_end_ce": ${SEMANTIC_END_WEIGHT},
    "semantic_end_margin": ${SEMANTIC_END_MARGIN_WEIGHT},
    "semantic_end_logit_margin": ${SEMANTIC_END_LOGIT_MARGIN},
    "semantic_rollin_end_ce": ${ROLLIN_END_WEIGHT},
    "semantic_rollin_continue_decision_margin": ${ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT},
    "semantic_rollin_continue_decision_logit_margin": ${ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN},
    "semantic_boundary_binary": ${BOUNDARY_BINARY_WEIGHT},
    "semantic_boundary_binary_logit_margin": ${BOUNDARY_BINARY_LOGIT_MARGIN},
    "semantic_boundary_rollin_rate": ${BOUNDARY_ROLLIN_RATE},
    "semantic_boundary_rollin_ramp_updates": ${BOUNDARY_ROLLIN_RAMP_UPDATES},
    "boundary_eos": ${BOUNDARY_EOS_WEIGHT},
    "asr_ce": 1.0, "mt_ce": 1.0, "semantic_ce": 1.0,
    "replay_ce": 0.50, "v1_asr_kl": 0.30, "phase3_kl": 0.25,
    "commit_consistency": 0.20
  },
  "tensorboard": "${RUN_TENSORBOARD_DIR}",
  "log": "${RUN_LOG}",
  "gpu_csv": "${RUN_LOG%.log}.gpu.csv",
  "frozen_stage_a_audit": "${FROZEN_AUDIT}",
  "continuation_input_audit": "${CONTINUATION_AUDIT}",
  "next_required_gate": "fixed_16_free_running_validation_with_local_agreement_and_pacing",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "rollin_continuation_status=complete"
echo "final_checkpoint=${FINAL_CHECKPOINT}"
echo "report=${RUN_SUMMARY}"
echo "tensorboard=${RUN_TENSORBOARD_DIR}"
