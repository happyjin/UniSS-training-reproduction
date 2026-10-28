#!/bin/bash
# Detached torchrun on the given cards.  launch_ddp.sh <cards e.g. 3,4> <port> <log> <script.py> [args...]
# PY_BIN (default uniss-train) selects the environment's bin directory.
set -euo pipefail
CARDS="${1:?cards}"; PORT="${2:?port}"; LOG="${3:?log}"; SCRIPT="${4:?script}"; shift 4
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
BIN="${PY_BIN:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin}"
N=$(awk -F, '{print NF}' <<< "$CARDS")
[ -e "$LOG" ] && { echo "refusing to overwrite $LOG"; exit 1; }
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$CARDS OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 setsid nohup "$BIN/torchrun" --nproc_per_node "$N" \
  --master_port "$PORT" "$SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
disown
echo "$SCRIPT on cards $CARDS -> $LOG"
