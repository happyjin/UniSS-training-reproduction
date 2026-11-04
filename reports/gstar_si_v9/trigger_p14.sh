#!/usr/bin/env bash
# P1-4: queue the speaker-blind seed trainings once no other user's process holds GPU memory for two checks 10 min apart
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
free_twice=0
while [ ! -e logs/multispk/v9/q/FINISH ]; do
  n=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    ps -o args= -p $p 2>/dev/null | grep -q "/opt/dlami/nvme/jasonleeeli/" || n=$((n+1))
  done
  if [ $n = 0 ]; then free_twice=$((free_twice+1)); else free_twice=0; fi
  if [ $free_twice -ge 2 ]; then cp -n reports/gstar_si_v9/jobs/p14_seeds.json logs/multispk/v9/q/040_p14_seeds.json; echo "P14_QUEUED $(date -u +%T)"; exit 0; fi
  sleep 600
done
