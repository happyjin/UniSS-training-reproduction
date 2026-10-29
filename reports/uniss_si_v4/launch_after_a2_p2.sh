#!/usr/bin/env bash
# After the A2 adapter has trained (run_v3_eval5.sh touches TRAINED) and the multi-speaker queue has placed its first
# jobs (5 min), start the stacked single-speaker queue P2 (A2 on dev, then RealSI); C1 runs earlier (launch_c1_now.sh).
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
until [ -f data/si_thinker_oracle_prefix_v7_spont/TRAINED ]; do sleep 60; done
sleep 300
for q in p2; do
  mkdir -p logs/si_v4/$q
  GPUQ_CARDS=0,1,2,3,4,5,6,7 setsid nohup /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq7.py \
    reports/uniss_si_v4/jobs/${q}_jobs.json logs/si_v4/$q > logs/si_v4/${q}_queue.log 2>&1 < /dev/null &
  sleep 5
done
echo "LAUNCHED p2 $(date -u +%H:%M)"
