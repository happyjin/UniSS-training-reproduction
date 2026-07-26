#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
SESSION="simul_uniss_v5_full198_mbs4_gbs128_stage3_8gpu"
LOG="${REPO_ROOT}/logs/simul_uniss_v5_full198_mbs4_gbs128_stage3/stage3_launcher.log"
SMOKE_MARKER="${REPO_ROOT}/runs/simul_uniss_v5_full198_mbs4_gbs128_stage3/stage03_mbs4_gbs128_smoke_v1/SMOKE_COMPLETE"
[[ -f "${SMOKE_MARKER}" ]] || { echo "MBS4 smoke incomplete: ${SMOKE_MARKER}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
mkdir -p "$(dirname "${LOG}")"
command="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${EXPERIMENT_DIR}/stage03_action_sft/run.sh") 2>&1 | tee -a $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
