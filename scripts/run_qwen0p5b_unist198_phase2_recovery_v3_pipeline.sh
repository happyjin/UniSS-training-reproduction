#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_phase2_recovery_v3.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

runner=("${REPO_ROOT}/scripts/run_qwen0p5b_unist198_phase2_recovery_v1.sh" --config "${CONFIG_FILE}")
gate_args=(
  --tensorboard-dir "${PILOT_TENSORBOARD_DIR}"
  --log "${PILOT_LOG_PATH}"
  --required-step "${PILOT_TRAIN_ITERS}"
  --max-valid-loss "${PILOT_MAX_VALID_LOSS}"
  --max-last-valid-loss "${PILOT_MAX_LAST_VALID_LOSS}"
  --grad-spike-threshold "${PILOT_GRAD_SPIKE_THRESHOLD}"
  --max-consecutive-grad-spikes "${PILOT_MAX_CONSECUTIVE_GRAD_SPIKES}"
  --absolute-max-grad-norm "${PILOT_ABSOLUTE_MAX_GRAD_NORM}"
  --output "${PILOT_RUN_DIR}/PILOT_GATE.json"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  "${runner[@]}" --mode pilot --dry-run
  printf '[dry-run] validate low-LR global-shuffle pilot through step %s; no early stop; absolute grad norm <= %s\n' \
    "${PILOT_TRAIN_ITERS}" "${PILOT_ABSOLUTE_MAX_GRAD_NORM}"
  printf '[dry-run] full continuation budget=%s; effective final iteration=%s\n' \
    "${FULL_TRAIN_ITERS}" "$((SOURCE_ITERATION + FULL_TRAIN_ITERS))"
  "${runner[@]}" --mode full --dry-run
  exit 0
fi

"${runner[@]}" --mode pilot
"${ENV_ROOT}/bin/python" -m training.validate_phase2_recovery "${gate_args[@]}"
printf 'passed_at=%s\n' "$(date -u +%FT%TZ)" > "${PILOT_RUN_DIR}/PILOT_GATE_PASSED"
echo "Recovery v3 pilot passed; starting isolated full low-LR global-shuffle continuation."
"${runner[@]}" --mode full
