#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?RUN_ID is required}"
: "${CANDIDATE_HF:?CANDIDATE_HF is required}"
SESSION=${SESSION:-${RUN_ID}}
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}
command=(env RUN_ID="${RUN_ID}" CANDIDATE_HF="${CANDIDATE_HF}")
for name in FORMAL_DATA_RUN_ID RUN_ROOT SELECTION CANDIDATE_FINGERPRINT GOLD \
  CANARY_REPORT CANDIDATE_CHECKPOINT BICODEC_MODEL NUM_WORKERS \
  MAX_S2S_SEMANTIC_TOKENS; do
  if [[ -n "${!name:-}" ]]; then
    command+=("${name}=${!name}")
  fi
done
command+=("${SCRIPT_DIR}/run_free_running_gate_8gpu.sh")
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" "cd $(printf %q "${REPO_ROOT}") && ${quoted}"
echo "started ${SESSION}"
