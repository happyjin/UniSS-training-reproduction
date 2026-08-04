#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
SESSION=${UNISS_STAGE13_SESSION:-uniss_streamspeech_stage13_public_v1}
GPU=${UNISS_STAGE13_GPU:-0}
PORT=${UNISS_STAGE13_PORT:-7865}
DIR=$ROOT/web_demo/uniss_streamspeech_stage13_v1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi
used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -ge 2048 ]; then
  echo "GPU $GPU is not idle: ${used} MiB" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "port in use: $PORT" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU' UNISS_STAGE13_DEVICE='cuda:0' UNISS_STAGE13_PORT='$PORT' '$DIR/watchdog.sh'"
echo "SESSION=$SESSION"
echo "GPU=$GPU"
echo "PORT=$PORT"
echo "PUBLIC_URL_FILE=$DIR/public_url.txt"
