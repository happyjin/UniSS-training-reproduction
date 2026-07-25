#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
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

[[ -f "${SMOKE_MARKER}" ]] || {
  echo "Global-shuffle smoke has not completed: ${SMOKE_MARKER}" >&2
  exit 1
}
mkdir -p "${PIPELINE_DIR}" "${LOG_DIR}"

for spec in "${stages[@]}"; do
  IFS=: read -r name root expected launcher <<< "${spec}"
  marker="${PIPELINE_DIR}/${name}.complete"
  if [[ -f "${marker}" ]]; then
    echo "Skipping completed ${name}"
    continue
  fi
  if [[ -e "${root}" ]]; then
    echo "Refusing to overwrite partial ${name} output: ${root}" >&2
    exit 1
  fi
  "${launcher}"
  [[ -f "${root}/latest_checkpointed_iteration.txt" ]] || {
    echo "${name}: missing checkpoint pointer" >&2
    exit 1
  }
  iteration="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")"
  [[ "${iteration}" == "${expected}" ]] || {
    echo "${name}: expected iteration ${expected}, got ${iteration}" >&2
    exit 1
  }
  printf 'completed_at=%s\niteration=%s\ncheckpoint=%s\n' \
    "$(date -u +%FT%TZ)" "${iteration}" "${root}" > "${marker}"
done

printf 'completed_at=%s\nanchor=%s\n' \
  "$(date -u +%FT%TZ)" "${QWEN_CHECKPOINT_ROOT}" > "${PIPELINE_MARKER}"
echo "Eight-GPU Qwen pipeline completed: ${PIPELINE_MARKER}"
