#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION="${WHISPER_V2_TMUX_SESSION:-whisper_attention_mask_v2_sharded}"
GPU_LIST_VALUE="${WHISPER_V2_GPU_LIST:-1,2,3,4,5,6,7}"
NUM_SHARDS="${WHISPER_V2_NUM_SHARDS:-2}"
MANIFEST="${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/runs.tsv"
IFS=',' read -r -a GPUS <<<"${GPU_LIST_VALUE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ "${#GPUS[@]}" -lt 1 || "${NUM_SHARDS}" -lt 1 ]]; then
  echo "Invalid GPU or shard configuration" >&2
  exit 2
fi

task_index=0
run_index=0
while IFS=$'\t' read -r label relative_root; do
  [[ -n "${label}" ]] || continue
  for ((shard_index=0; shard_index<NUM_SHARDS; shard_index++)); do
    gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
    command="cd '${REPO_ROOT}' && '${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/run_shard.sh' '${label}' '${REPO_ROOT}/${relative_root}' '${gpu}' '${shard_index}' '${NUM_SHARDS}'"
    window="r${run_index}s${shard_index}g${gpu}"
    if [[ "${task_index}" == "0" ]]; then
      tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
    else
      tmux new-window -t "${SESSION}" -n "${window}" "${command}"
    fi
    task_index=$((task_index + 1))
  done
  run_index=$((run_index + 1))
done < "${MANIFEST}"

echo "SESSION=${SESSION}"
echo "GPUS=${GPU_LIST_VALUE}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "TASKS=${task_index}"
