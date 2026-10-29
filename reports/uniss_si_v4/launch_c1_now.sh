#!/usr/bin/env bash
# The review controls (C1) now, stacked beside the multi-speaker evaluation by gpuq7, instead of after A2 has trained;
# then the v4 RealSI text tables with them (realsi_text.py; the file it rewrites is copied aside first).
# Pair with launch_after_a2_p2.sh, which starts only P2 after A2. Launch detached (setsid nohup); check with bash -n.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
G=paper/bayes_simul_s2st_v4/generated
[ -e logs/si_v4/c1 ] && { echo "logs/si_v4/c1 exists; C1 already launched"; exit 1; }
mkdir -p logs/si_v4/c1
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq7.py \
  reports/uniss_si_v4/jobs/c1_jobs.json logs/si_v4/c1
[ -e $G/realsi_text.pre_c1.json ] || cp -p $G/realsi_text.json $G/realsi_text.pre_c1.json
PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u paper/bayes_simul_s2st_v4/scripts/realsi_text.py
echo "C1_DONE $(date -u +%H:%M)"
