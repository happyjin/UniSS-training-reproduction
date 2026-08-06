#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

SESSION="${SESSION:-uniss_phase3_joint_full198_stage_a_high_throughput}"
GPU_COUNT="${GPU_COUNT:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
TOTAL_WORKERS=$((GPU_COUNT * WORKERS_PER_GPU))
if (( GPU_COUNT <= 0 || WORKERS_PER_GPU <= 0 )); then
  echo "GPU_COUNT and WORKERS_PER_GPU must be positive" >&2
  exit 1
fi
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session exists: ${SESSION}" >&2
  exit 1
}
if pgrep -f '[s]tage_a prepare-part.*full198_stage_a/parts' >/dev/null; then
  echo "existing full198 Stage-A prepare-part process detected" >&2
  exit 1
fi

LOG_ROOT="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v1/full198_stage_a_high_throughput"
mkdir -p "${FORMAL_STAGE_A_ROOT}/parts" "${LOG_ROOT}"
tmux new-session -d -s "${SESSION}" -n worker0
for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  (( worker == 0 )) || tmux new-window -t "${SESSION}" -n "worker${worker}"
  gpu=$((worker % GPU_COUNT))
  log="${LOG_ROOT}/worker${worker}_gpu${gpu}.log"
  command="cd ${REPO_ROOT} && bash ${SCRIPT_ROOT}/prepare_full198_stage_a_parallel_worker.sh ${worker} ${TOTAL_WORKERS} ${gpu} 2>&1 | tee ${log}"
  tmux send-keys -t "${SESSION}:worker${worker}" "${command}" C-m
done
echo "Started ${SESSION}: ${TOTAL_WORKERS} workers, ${WORKERS_PER_GPU} per GPU."
