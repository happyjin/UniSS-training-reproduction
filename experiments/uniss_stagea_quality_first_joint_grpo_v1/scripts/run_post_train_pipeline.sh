#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"

POST_ID=${POST_ID:-formal_complete_v1}
OUTPUT_ROOT=${EVAL_ROOT}/${POST_ID}
FINAL_REPORT_ROOT=${REPORT_ROOT}/${POST_ID}
PIPELINE_LOG=${LOG_ROOT}/${POST_ID}.post_train.log
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

ARMS=(
  a1_sft_full_recovery1
  a2_g4_full_recovery1
  a3_g8_full_recovery1
  a4_g8_seed2_full_recovery1
)

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 3; }
}

for path in \
  "${EXPERIMENT_ROOT}/evaluation/training_audit.py" \
  "${EXPERIMENT_ROOT}/evaluation/compare_arms.py" \
  "${EXPERIMENT_ROOT}/evaluation/write_report.py" \
  "${EXPERIMENT_ROOT}/evaluation/protocols/validation64_e2e16_seed20260825.json" \
  "${EXPERIMENT_ROOT}/evaluation/protocols/long_audio4_prefix60.json" \
  "${EXPERIMENT_ROOT}/evaluation/protocols/long_audio4_full.json" \
  "${EXPERIMENT_ROOT}/scripts/run_routed_eval.sh" \
  "${EXPERIMENT_ROOT}/scripts/run_short_audio_suite.sh" \
  "${EXPERIMENT_ROOT}/scripts/run_long_prefix_suite.sh" \
  "${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  "${PYTHON_BIN}"; do
  require_file "${path}"
done
for arm in "${ARMS[@]}"; do
  require_file "${LOG_ROOT}/${arm}.log"
  require_file "${LOG_ROOT}/${arm}.gpu.csv"
done

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "POST_ID=${POST_ID}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "FINAL_REPORT_ROOT=${FINAL_REPORT_ROOT}"
  echo "CHECK=passed"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${FINAL_REPORT_ROOT}" "${LOG_ROOT}" "${USER_ROOT}/tmp"
exec 9>"${USER_ROOT}/tmp/${EXPERIMENT_NAME}_${POST_ID}.lock"
flock -n 9 || { echo "post-train pipeline is already running" >&2; exit 4; }
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

checkpoint_ready() {
  local arm=$1 latest=${CHECKPOINT_ROOT}/${arm}/latest_checkpointed_iteration.txt
  [[ -f "${latest}" ]] || return 1
  [[ "$(tr -d '[:space:]' < "${latest}")" == "2510" ]] || return 1
  [[ -f "${CHECKPOINT_ROOT}/${arm}/iter_0002510/.metadata" ]]
}

all_checkpoints_ready() {
  local arm
  for arm in "${ARMS[@]}"; do
    checkpoint_ready "${arm}" || return 1
  done
}

log "waiting for all four immutable iter_0002510 checkpoints"
while ! all_checkpoints_ready; do
  states=()
  for arm in "${ARMS[@]}"; do
    latest=${CHECKPOINT_ROOT}/${arm}/latest_checkpointed_iteration.txt
    step=0
    [[ -f "${latest}" ]] && step=$(tr -d '[:space:]' < "${latest}")
    states+=("${arm}:${step}")
  done
  log "checkpoint progress ${states[*]}"
  sleep 30
done

# Megatron saves iter_0002510 before completing its final validation. Do not
# contend for the same devices until every training process has released them.
while pgrep -f "${EXPERIMENT_ROOT}/training/pretrain_megatron.py" >/dev/null; do
  log "final checkpoints exist; waiting for Megatron final validation to exit"
  sleep 30
done
log "training complete; starting fixed post-training evaluation"

TRAINING_AUDIT=${FINAL_REPORT_ROOT}/TRAINING_AUDIT.json
"${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/training_audit.py" \
  --arm "${ARMS[0]}=${LOG_ROOT}/${ARMS[0]}.log=${LOG_ROOT}/${ARMS[0]}.gpu.csv=0,1" \
  --arm "${ARMS[1]}=${LOG_ROOT}/${ARMS[1]}.log=${LOG_ROOT}/${ARMS[1]}.gpu.csv=2,3" \
  --arm "${ARMS[2]}=${LOG_ROOT}/${ARMS[2]}.log=${LOG_ROOT}/${ARMS[2]}.gpu.csv=4,5" \
  --arm "${ARMS[3]}=${LOG_ROOT}/${ARMS[3]}.log=${LOG_ROOT}/${ARMS[3]}.gpu.csv=6,7" \
  --output "${TRAINING_AUDIT}" > "${FINAL_REPORT_ROOT}/training_audit.stdout.json"

ROUTED_ROOT=${OUTPUT_ROOT}/routed64_e2e16
mkdir -p "${ROUTED_ROOT}"
for arm in "${ARMS[@]}"; do
  log "routed validation64/E2E16: ${arm} on 8 GPUs"
  SELECTION=${EXPERIMENT_ROOT}/evaluation/protocols/validation64_e2e16_seed20260825.json \
    "${EXPERIMENT_ROOT}/scripts/run_routed_eval.sh" \
    "${arm}" \
    "${CHECKPOINT_ROOT}/${arm}/iter_0002510" \
    "${ROUTED_ROOT}/${arm}" 8
done

COMPARISON=${FINAL_REPORT_ROOT}/COMPARISON.json
BEST_ARM_FILE=${FINAL_REPORT_ROOT}/BEST_ARM.txt
compare=(
  "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/compare_arms.py"
  --output "${COMPARISON}"
  --best-output "${BEST_ARM_FILE}"
)
for arm in "${ARMS[@]}"; do
  compare+=(--arm "${arm}=${ROUTED_ROOT}/${arm}/SUMMARY.json")
done
"${compare[@]}" > "${FINAL_REPORT_ROOT}/comparison.stdout.json"
BEST_ARM=$(tr -d '[:space:]' < "${BEST_ARM_FILE}")
log "quality-first selected arm: ${BEST_ARM}"

SHORT_ROOT=${OUTPUT_ROOT}/short_audio_multichunk
mkdir -p "${SHORT_ROOT}"
pids=()
for index in 0 1 2 3; do
  arm=${ARMS[$index]}
  gpu0=$((index * 2))
  gpu1=$((gpu0 + 1))
  "${EXPERIMENT_ROOT}/scripts/run_short_audio_suite.sh" \
    "${arm}" "${CHECKPOINT_ROOT}/${arm}/iter_0002510" \
    "${SHORT_ROOT}/${arm}" "${gpu0}" "${gpu1}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || { log "short audio evaluation failed"; exit 1; }

PREFIX_ROOT=${OUTPUT_ROOT}/long_audio4_prefix60_multichunk
mkdir -p "${PREFIX_ROOT}"
for offset in 0 2; do
  arm0=${ARMS[$offset]}
  arm1=${ARMS[$((offset + 1))]}
  "${EXPERIMENT_ROOT}/scripts/run_long_prefix_suite.sh" \
    "${arm0}" "${CHECKPOINT_ROOT}/${arm0}/iter_0002510" \
    "${PREFIX_ROOT}/${arm0}" 0 1 2 3 & p0=$!
  "${EXPERIMENT_ROOT}/scripts/run_long_prefix_suite.sh" \
    "${arm1}" "${CHECKPOINT_ROOT}/${arm1}/iter_0002510" \
    "${PREFIX_ROOT}/${arm1}" 4 5 6 7 & p1=$!
  status=0
  wait "${p0}" || status=1
  wait "${p1}" || status=1
  [[ "${status}" -eq 0 ]] || { log "long prefix evaluation failed"; exit 1; }
done

LONG_ROOT=${OUTPUT_ROOT}/bounded_longform_chunk640
BEST_LONG=${LONG_ROOT}/${BEST_ARM}
STAGE_A_LONG=${LONG_ROOT}/stage_a_iter381
mkdir -p "${LONG_ROOT}"
"${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  "${BEST_ARM}" "${CHECKPOINT_ROOT}/${BEST_ARM}/iter_0002510" \
  "${BEST_LONG}" 0 1 2 3 640 & p0=$!
"${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  stage_a_iter381 NONE "${STAGE_A_LONG}" 4 5 6 7 640 & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
[[ "${status}" -eq 0 ]] || { log "bounded long-form evaluation failed"; exit 1; }

FINAL_REPORT=${FINAL_REPORT_ROOT}/REPORT.zh-CN.md
"${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/write_report.py" \
  --evaluation-root "${ROUTED_ROOT}" \
  --training-audit "${TRAINING_AUDIT}" \
  --comparison "${COMPARISON}" \
  --short-root "${SHORT_ROOT}" \
  --long-prefix-root "${PREFIX_ROOT}" \
  --best-longform "${BEST_LONG}" \
  --stage-a-longform "${STAGE_A_LONG}" \
  --output "${FINAL_REPORT}"

echo "complete" > "${FINAL_REPORT_ROOT}/PIPELINE_COMPLETE"
log "post-training pipeline complete: ${FINAL_REPORT}"
