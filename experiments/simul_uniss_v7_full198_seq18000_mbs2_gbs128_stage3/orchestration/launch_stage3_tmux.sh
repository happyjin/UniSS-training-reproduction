#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="simul_uniss_v7_seq18000_stage3_8gpu"
LOG="${LOG_DIR}/stage3_launcher.log"
SMOKE_MARKER="${RUN_DIR}/stage03_seq18000_mbs2_gbs128_smoke_v1/SMOKE_COMPLETE"
[[ -f "${FULL_DATA_READY_MARKER}" && -f "${SMOKE_MARKER}" ]] || { echo "Data or smoke not ready" >&2; exit 1; }
[[ -f "${STAGE3_LOAD_ROOT}/latest_checkpointed_iteration.txt" ]] || { echo "Missing load checkpoint" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
[[ ! -e "${STAGE3_SAVE_ROOT}" && ! -e "${STAGE3_TENSORBOARD_DIR}" && ! -e "${LOG}" ]] || {
  echo "Refusing to overwrite v7 output" >&2; exit 1;
}
mkdir -p "${LOG_DIR}"
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/stage03_action_sft/run.sh") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
