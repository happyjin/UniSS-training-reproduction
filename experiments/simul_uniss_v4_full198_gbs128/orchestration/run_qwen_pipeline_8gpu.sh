#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
for variable in STAGE3_TRAIN_ITERS STAGE4_TRAIN_ITERS STAGE6_TRAIN_ITERS; do
  [[ -n "${!variable:-}" ]] || { echo "Missing ${variable}; generate ${TRAINING_SCHEDULE_FILE}" >&2; exit 1; }
done
PIPELINE_DIR="${RUN_DIR}/qwen_pipeline_8gpu"
SMOKE_MARKER="${RUN_DIR}/shuffle_smoke_8gpu_gbs128_v1/SHUFFLE_SMOKE_COMPLETE"
PIPELINE_MARKER="${PIPELINE_DIR}/QWEN_PIPELINE_COMPLETE"
stages=(
  "stage03_action_sft:${STAGE3_SAVE_ROOT}:${STAGE3_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage03_action_sft/run.sh"
  "stage04_interleaved_s2st:${STAGE4_SAVE_ROOT}:${STAGE4_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage04_interleaved_s2st/run.sh"
  "stage06_joint_refinement:${STAGE6_SAVE_ROOT}:${STAGE6_TRAIN_ITERS}:${EXPERIMENT_DIR}/stage06_joint_refinement/run.sh"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  for spec in "${stages[@]}"; do
    IFS=: read -r name root expected launcher <<< "${spec}"
    echo "stage=${name} output=${root} expected_iteration=${expected}"
    "${launcher}" --dry-run
  done
  exit 0
fi
[[ -f "${FULL_DATA_READY_MARKER}" ]] || { echo "Full data incomplete" >&2; exit 1; }
[[ -f "${SMOKE_MARKER}" ]] || { echo "GBS128 smoke incomplete: ${SMOKE_MARKER}" >&2; exit 1; }
mkdir -p "${PIPELINE_DIR}" "${LOG_DIR}"
for spec in "${stages[@]}"; do
  IFS=: read -r name root expected launcher <<< "${spec}"
  marker="${PIPELINE_DIR}/${name}.complete"
  if [[ -f "${marker}" ]]; then
    [[ -f "${root}/latest_checkpointed_iteration.txt" ]] || {
      echo "Completed marker has no checkpoint pointer: ${marker}" >&2; exit 1;
    }
    actual="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")"
    [[ "${actual}" == "${expected}" ]] || {
      echo "Completed ${name}: expected ${expected}, got ${actual}" >&2; exit 1;
    }
    echo "Skipping verified completed ${name}"
    continue
  fi
  [[ ! -e "${root}" ]] || { echo "Refusing existing stage output: ${root}" >&2; exit 1; }
  "${launcher}"
  actual="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")"
  [[ "${actual}" == "${expected}" ]] || { echo "${name}: expected ${expected}, got ${actual}" >&2; exit 1; }
  printf 'completed_at=%s\niteration=%s\ncheckpoint=%s\n' \
    "$(date -u +%FT%TZ)" "${actual}" "${root}" > "${marker}"
done
printf 'completed_at=%s\nanchor=%s\nmicro_batch=%s\nglobal_batch=%s\n' \
  "$(date -u +%FT%TZ)" "${QWEN_CHECKPOINT_ROOT}" "${SIMUL_MICRO_BATCH_SIZE}" \
  "${SIMUL_GLOBAL_BATCH_SIZE}" > "${PIPELINE_MARKER}"
