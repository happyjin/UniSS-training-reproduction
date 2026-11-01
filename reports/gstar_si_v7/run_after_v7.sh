#!/usr/bin/env bash
# Paper v7: run after_v7.sh every 5 minutes until logs/multispk/v7/fu/ALL exists or logs/multispk/v7/fu/STOP is made.
# after_v7.sh is re-read at every round, so it can be replaced (write a new file, then command mv -f) while this runs.
#   setsid nohup bash reports/gstar_si_v7/run_after_v7.sh > /dev/null 2>&1 < /dev/null &
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
F=logs/multispk/v7/fu; mkdir -p $F
exec 8> $F/after.lock
flock -n 8 || exit 1
while [ ! -e $F/ALL ] && [ ! -e $F/STOP ]; do
  bash reports/gstar_si_v7/after_v7.sh >> logs/multispk/v7/after.log 2>&1
  sleep 300
done
