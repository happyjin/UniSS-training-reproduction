#!/usr/bin/env bash
set -uo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
DIR=$ROOT/web_demo/uniss_streamspeech_stage13_v1
mkdir -p "$DIR/runtime_logs"
while true; do
  rm -f "$DIR/public_url.txt" "$DIR/access_info.json"
  echo "[$(date -Is)] starting Stage13 public research demo" | tee -a "$DIR/runtime_logs/watchdog.log"
  "$DIR/run_public.sh" 2>&1 | tee -a "$DIR/runtime_logs/public_server.log"
  code=${PIPESTATUS[0]}
  echo "[$(date -Is)] exited code=$code; restart in 30 seconds" | tee -a "$DIR/runtime_logs/watchdog.log"
  sleep 30
done
