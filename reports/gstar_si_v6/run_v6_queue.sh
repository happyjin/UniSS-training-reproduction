#!/usr/bin/env bash
# Paper v6 (MIST) GPU queue: a gpuq7 queue for one job file, beside the single-speaker branch's queues (shared ledger
# logs/gpuq7_ledger: each queue counts the other's reservations as its own, so the two never stack beyond OUR_MAX on a
# card). Every card now carries a colleague's job (13-22 GB), hence CAP 126,000 MiB; OUR_MAX 105,000 MiB allows one of
# our ~100 GB decode jobs per card and nothing of ours beside it. The memory watchdog (mem_watchdog.py) runs while any
# v6 queue runs. Usage:
#   setsid nohup bash reports/gstar_si_v6/run_v6_queue.sh <jobs.json> <logdir> > /dev/null 2>&1 < /dev/null &
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
JOBS=$1; LD=$2
mkdir -p "$LD"
touch "$LD/allow_empty"   # cards with only a small colleague job still count
L=logs/multispk/v6
if ! { [ -f $L/watchdog.pid ] && kill -0 "$(cat $L/watchdog.pid)" 2>/dev/null; }; then
  setsid nohup python3 reports/gstar_si_v6/mem_watchdog.py --until-file $L/WATCHDOG_STOP >> $L/watchdog.log 2>&1 < /dev/null &
  echo $! > $L/watchdog.pid
fi
echo "[$(date -u '+%F %T') UTC] MIST(v6): gpuq7 queue $JOBS (logs $LD), CAP 126000, OUR_MAX 105000, watchdog on" >> logs/gpu_coord/events.log
GPUQ_CARDS=0,1,2,3,4,5,6,7 CAP=126000 OUR_MAX=105000 PIGGY_MIN=0 \
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq7.py "$JOBS" "$LD" >> "$LD/queue.log" 2>&1
echo "[$(date -u '+%F %T') UTC] MIST(v6): queue $JOBS finished: $(tail -n 1 $LD/queue.log)" >> logs/gpu_coord/events.log
