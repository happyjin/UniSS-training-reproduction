#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"

SESSION="simul_uniss_v3_full198_training_waiter"
DATA_SESSION="simul_uniss_v3_full198_data"
QWEN_SESSION="simul_uniss_v3_full198_qwen_8gpu"
COMPONENT_SESSION="simul_uniss_v3_full198_components_8gpu"
SMOKE_MARKER="${RUN_DIR}/shuffle_smoke_8gpu_v1/SHUFFLE_SMOKE_COMPLETE"
QWEN_MARKER="${RUN_DIR}/qwen_pipeline_8gpu/QWEN_PIPELINE_COMPLETE"
COMPONENT_MARKER="${RUN_DIR}/component_pipeline_8gpu/COMPONENT_PIPELINE_COMPLETE"
LOG="${LOG_DIR}/training_when_ready_launcher.log"

wait_and_launch="$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${REPO_ROOT}")
while [[ ! -f $(printf '%q' "${FULL_DATA_READY_MARKER}") ]]; do
  if ! tmux has-session -t $(printf '%q' "${DATA_SESSION}") 2>/dev/null; then
    echo 'Full-data preparation ended without FULL_DATA_READY.json' >&2
    exit 1
  fi
  complete=\$(find $(printf '%q' "${PACKED_PARTS_DIR}") -maxdepth 2 -name PACK_COMPLETE.json 2>/dev/null | wc -l)
  echo "\$(date -u +%FT%TZ) waiting for full data: \${complete}/${SHARD_COUNT} packed"
  sleep 60
done
echo "\$(date -u +%FT%TZ) full data ready: $(printf '%q' "${FULL_DATA_READY_MARKER}")"
if [[ ! -f $(printf '%q' "${SMOKE_MARKER}") ]]; then
  $(printf '%q' "${EXPERIMENT_DIR}/orchestration/run_shuffle_smoke_8gpu.sh")
else
  echo 'Skipping completed eight-GPU shuffle smoke'
fi
if ! tmux has-session -t $(printf '%q' "${TENSORBOARD_SESSION}") 2>/dev/null; then
  $(printf '%q' "${EXPERIMENT_DIR}/orchestration/start_tensorboard.sh")
else
  echo 'TensorBoard session already exists'
fi
if [[ ! -f $(printf '%q' "${QWEN_MARKER}") ]]; then
  if tmux has-session -t $(printf '%q' "${QWEN_SESSION}") 2>/dev/null; then
    echo 'Qwen pipeline session already exists'
  else
    $(printf '%q' "${EXPERIMENT_DIR}/orchestration/launch_qwen_pipeline_tmux.sh")
  fi
else
  echo 'Skipping completed Qwen pipeline'
fi
if [[ ! -f $(printf '%q' "${COMPONENT_MARKER}") ]]; then
  if tmux has-session -t $(printf '%q' "${COMPONENT_SESSION}") 2>/dev/null; then
    echo 'Component pipeline session already exists'
  else
    $(printf '%q' "${EXPERIMENT_DIR}/orchestration/launch_component_pipeline_when_ready.sh")
  fi
else
  echo 'Skipping completed component pipeline'
fi
echo "\$(date -u +%FT%TZ) full198 training launch sequence handed off"
EOF
)"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'session=%s\nlog=%s\nready_marker=%s\n' "${SESSION}" "${LOG}" "${FULL_DATA_READY_MARKER}"
  printf '%s\n' "${wait_and_launch}"
  exit 0
fi

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
}
[[ -f "${FULL_DATA_READY_MARKER}" ]] || tmux has-session -t "${DATA_SESSION}" 2>/dev/null || {
  echo "Neither full data nor its preparation session is available" >&2
  exit 1
}
mkdir -p "$(dirname "${LOG}")"
printf -v command '{\n%s\n} 2>&1 | tee -a %q' "${wait_and_launch}" "${LOG}"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "Started ${SESSION}; log=${LOG}"
