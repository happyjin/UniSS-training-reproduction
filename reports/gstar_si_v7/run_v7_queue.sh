#!/usr/bin/env bash
# Paper v7 (MIST): the branch's ONE GPU queue (gpuq7v7.py, jobs added later through <logdir>/add/*.json) and its memory
# watchdog (mem_watchdog_v7.py). Every card carries a colleague's job (~50-55 GB since 2026-11-01): CAP 120,000 MiB
# per card (no job starts that would bring a card above it), OUR_MAX 70,000 MiB of ours per card. A lock keeps a
# second copy from starting. Launch (detached):
#   setsid nohup bash reports/gstar_si_v7/run_v7_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/multispk/v7; Q=$L/queue
mkdir -p $Q/add
exec 9> $L/queue.lock
flock -n 9 || { echo "v7 queue already running"; exit 1; }
if ! { [ -f $L/watchdog.pid ] && kill -0 "$(cat $L/watchdog.pid)" 2>/dev/null; }; then
  setsid nohup python3 reports/gstar_si_v7/mem_watchdog_v7.py --until-file $L/WATCHDOG_STOP >> $L/watchdog.log 2>&1 < /dev/null &
  echo $! > $L/watchdog.pid
fi
echo "[$(date -u '+%F %T') UTC] MIST(v7): single gpuq7v7 queue $Q started (CAP ${CAP:-120000}, OUR_MAX ${OUR_MAX:-70000}), watchdog v7 pid $(cat $L/watchdog.pid)" >> logs/gpu_coord/events.log
GPUQ_CARDS=${CARDS:-0,1,2,3,4,5,6,7} CAP=${CAP:-120000} OUR_MAX=${OUR_MAX:-70000} \
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u reports/gstar_si_v7/gpuq7v7.py $Q >> $Q/queue.log 2>&1
echo "[$(date -u '+%F %T') UTC] MIST(v7): queue finished: $(tail -n 1 $Q/queue.log)" >> logs/gpu_coord/events.log
