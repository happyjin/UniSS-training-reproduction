#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RECOVER_COMPLETED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --recover-completed) RECOVER_COMPLETED=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

PIPELINE_DIR="${RUN_DIR}/qwen_pipeline_8gpu"
SMOKE_NAME="${SHUFFLE_SMOKE_NAME:-shuffle_smoke_8gpu_v2}"
SMOKE_MARKER="${RUN_DIR}/${SMOKE_NAME}/SHUFFLE_SMOKE_COMPLETE"
PIPELINE_MARKER="${PIPELINE_DIR}/QWEN_PIPELINE_COMPLETE"

stages=(
  "stage03_action_sft:${STAGE3_SAVE_ROOT}:${STAGE3_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage03_action_sft/run.sh:stage_action_qwen.log"
  "stage04_interleaved_s2st:${STAGE4_SAVE_ROOT}:${STAGE4_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh:stage_interleaved_qwen.log"
  "stage06_joint_refinement:${STAGE6_SAVE_ROOT}:${STAGE6_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh:stage_joint_qwen.log"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  for spec in "${stages[@]}"; do
    IFS=: read -r name root expected launcher log_name <<< "${spec}"
    echo "stage=${name} output=${root} expected_iteration=${expected}"
    "${launcher}" --dry-run
  done
  exit 0
fi

[[ -f "${SMOKE_MARKER}" ]] || {
  echo "Global-shuffle smoke has not completed: ${SMOKE_MARKER}" >&2
  exit 1
}
mkdir -p "${PIPELINE_DIR}" "${LOG_DIR}"

verify_completed_stage() {
  local name="$1" root="$2" expected="$3" log_file="$4"
  local actual shard_count
  [[ -f "${root}/latest_checkpointed_iteration.txt" ]] || {
    echo "${name}: missing checkpoint pointer" >&2
    return 1
  }
  actual="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${name}: expected iteration ${expected}, got ${actual}" >&2
    return 1
  }
  [[ -d "${root}/iter_$(printf '%07d' "${expected}")" ]] || {
    echo "${name}: missing iteration directory ${expected}" >&2
    return 1
  }
  shard_count="$(find "${root}/iter_$(printf '%07d' "${expected}")" -maxdepth 1 -type f -name '__*_0.distcp' | wc -l)"
  (( shard_count >= SIMUL_NPROC_PER_NODE )) || {
    echo "${name}: expected at least ${SIMUL_NPROC_PER_NODE} distributed checkpoint shards, got ${shard_count}" >&2
    return 1
  }
  [[ -f "${log_file}" ]] || { echo "${name}: missing log ${log_file}" >&2; return 1; }
  rg -q "iteration +${expected}/ +${expected}" "${log_file}"
  rg -q "validation loss at iteration +${expected}" "${log_file}"
  if rg -q 'number of skipped iterations:[ ]+[1-9]|number of nan iterations:[ ]+[1-9]' "${log_file}"; then
    echo "${name}: nonzero skipped/NaN iteration found in ${log_file}" >&2
    return 1
  fi
}

write_stage_marker() {
  local marker="$1" iteration="$2" root="$3" recovery="$4" log_file="$5"
  local temporary="${marker}.tmp.$$"
  printf 'completed_at=%s\niteration=%s\ncheckpoint=%s\nrecovered=%s\nverified_log=%s\n' \
    "$(date -u +%FT%TZ)" "${iteration}" "${root}" "${recovery}" "${log_file}" > "${temporary}"
  mv "${temporary}" "${marker}"
}

for spec in "${stages[@]}"; do
  IFS=: read -r name root expected launcher log_name <<< "${spec}"
  marker="${PIPELINE_DIR}/${name}.complete"
  log_file="${LOG_DIR}/${log_name}"
  if [[ -f "${marker}" ]]; then
    verify_completed_stage "${name}" "${root}" "${expected}" "${log_file}"
    echo "Skipping completed ${name}"
    continue
  fi
  if [[ -e "${root}" ]]; then
    if [[ "${RECOVER_COMPLETED}" == "1" ]]; then
      verify_completed_stage "${name}" "${root}" "${expected}" "${log_file}"
      write_stage_marker "${marker}" "${expected}" "${root}" 1 "${log_file}"
      echo "Recovered verified completed ${name}"
      continue
    fi
    echo "Refusing to overwrite existing ${name} output without --recover-completed: ${root}" >&2
    exit 1
  fi
  status=0
  "${launcher}" || status=$?
  if (( status != 0 )) && [[ "${RECOVER_COMPLETED}" != "1" ]]; then
    echo "${name}: launcher failed with status ${status}" >&2
    exit "${status}"
  fi
  verify_completed_stage "${name}" "${root}" "${expected}" "${log_file}"
  write_stage_marker "${marker}" "${expected}" "${root}" "$((status != 0))" "${log_file}"
done

printf 'completed_at=%s\nanchor=%s\n' \
  "$(date -u +%FT%TZ)" "${QWEN_CHECKPOINT_ROOT}" > "${PIPELINE_MARKER}"
echo "Eight-GPU Qwen pipeline completed: ${PIPELINE_MARKER}"
