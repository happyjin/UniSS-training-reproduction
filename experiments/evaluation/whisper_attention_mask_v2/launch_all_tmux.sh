#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION="${WHISPER_V2_TMUX_SESSION:-whisper_attention_mask_v2}"
GPU_LIST_VALUE="${WHISPER_V2_GPU_LIST:-1,1,2,2,3,3,4,4,5,5,6,6,7,7}"
IFS=',' read -r -a GPUS <<<"${GPU_LIST_VALUE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "No GPUs configured" >&2
  exit 2
fi

for slot in "${!GPUS[@]}"; do
  gpu="${GPUS[${slot}]}"
  command="cd '${REPO_ROOT}' && '${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/run_worker.sh' '${gpu}' '${slot}' '${#GPUS[@]}'"
  if [[ "${slot}" == "0" ]]; then
    tmux new-session -d -s "${SESSION}" -n "gpu${gpu}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "gpu${gpu}" "${command}"
  fi
done

echo "SESSION=${SESSION}"
echo "GPUS=${GPU_LIST_VALUE}"
echo "RUNS=$(wc -l < "${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/runs.tsv")"
