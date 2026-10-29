#!/usr/bin/env bash
# P2's test_off_a2_en2zh failed at engine start (21:49 UTC): a D1b engine started on the same cards 12 s later and took
# the KV-cache memory. Re-run it alone once every other P2 job has started (so no two engines start together), stacked
# by gpuq7 on any cards with room. Launch detached; check with bash -n.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
for id in devsp_a2_en2zh devun_a2_en2zh devsp_a2_zh2en devun_a2_zh2en test_grid_a2_q_en2zh test_grid_a2_noq_en2zh test_off_a2_zh2en test_grid_a2_q_zh2en test_grid_a2_noq_zh2en; do
  until grep -qE " (start|skip) ${id}( |$)" logs/si_v4/p2_queue.log; do sleep 30; done
done
sleep 150   # the last P2 engine has profiled its memory
mkdir -p logs/si_v4/p2fix && touch logs/si_v4/p2fix/allow_empty
OUR_MAX=120000 GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq7.py \
  reports/uniss_si_v4/jobs/p2fix_jobs.json logs/si_v4/p2fix
