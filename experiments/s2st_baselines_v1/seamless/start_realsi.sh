#!/usr/bin/env bash
# R1: start the registered SeamlessStreaming RealSI grid (paper/baselines_run_plan_seamless_v1.md) through realsi_queue.py,
# detached. DRY_RUN=1 prints the job commands only. USE_TRAIN=1 lets the queue stack on cards 0-3 under its guard.
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
G=experiments/s2st_baselines_v1/seamless/grid_v1.json
[ -f "$G" ] || { echo "no grid $G"; exit 1; }
ARGS="--grid $G"
[ "${USE_TRAIN:-0}" = 1 ] && ARGS="$ARGS --use-train-cards"
mkdir -p paper/baselines_results/seamless logs/s2st_baselines_v1/realsi
if [ "${DRY_RUN:-0}" = 1 ]; then
  PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python experiments/s2st_baselines_v1/seamless/realsi_queue.py $ARGS --dry-run
  exit 0
fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
setsid nohup env PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u \
  experiments/s2st_baselines_v1/seamless/realsi_queue.py $ARGS > logs/s2st_baselines_v1/realsi_queue_stdout_$TS.log 2>&1 < /dev/null &
echo "queue pid $! log logs/s2st_baselines_v1/realsi_queue_stdout_$TS.log"
