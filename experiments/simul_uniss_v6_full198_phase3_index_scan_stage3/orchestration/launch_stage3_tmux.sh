#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
SESSION="simul_uniss_v6_full198_phase3_index_scan_stage3_8gpu"
LOG="${LOG_DIR}/stage3_launcher.log"
[[ -f "${FULL_DATA_READY_MARKER}" ]] || { echo "Missing ${FULL_DATA_READY_MARKER}" >&2; exit 1; }
[[ "$(tr -d '[:space:]' < "${V5_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")" == "500" ]] || {
  echo "Expected complete v5 iteration-500 checkpoint" >&2; exit 1;
}
[[ "$(find "${V5_CHECKPOINT_ROOT}/iter_0000500" -maxdepth 1 -name '*.distcp' | wc -l)" == "8" ]] || {
  echo "Expected eight v5 distributed checkpoint shards" >&2; exit 1;
}
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
[[ ! -e "${STAGE3_SAVE_ROOT}" && ! -e "${STAGE3_TENSORBOARD_DIR}" && ! -e "${LOG}" ]] || {
  echo "Refusing to overwrite an existing v6 output" >&2; exit 1;
}
mkdir -p "${LOG_DIR}"
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/stage03_action_sft/run.sh") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
