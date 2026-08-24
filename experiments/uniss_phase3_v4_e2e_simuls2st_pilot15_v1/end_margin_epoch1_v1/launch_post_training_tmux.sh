#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TRAIN_RUN_ID:?set the immutable END-margin training RUN_ID}"

SESSION=${SESSION:-endmargin_epoch1_post}
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}

command=(env TRAIN_RUN_ID="${TRAIN_RUN_ID}")
for name in DATA_RUN_ID TRAIN_ITERS POLL_SECONDS TRAIN_SESSION \
  MAX_S2S_SEMANTIC_TOKENS CANDIDATE_HF GATE_RUN_ID SELECTION POST_LOG; do
  if [[ -n "${!name:-}" ]]; then
    command+=("${name}=${!name}")
  fi
done
command+=("${HERE}/wait_then_export_gate.sh")
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" "cd $(printf %q "${HERE}") && ${quoted}"
echo "started ${SESSION} for ${TRAIN_RUN_ID}"
