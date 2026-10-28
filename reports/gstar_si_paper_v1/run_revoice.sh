#!/usr/bin/env bash
# One-voice-per-speaker Chinese AMI speech on all eight cards (make_revoice_jobs.py, gpuq6). The holders are already off
# (stopped for the diarizer runs); they come back at the end. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/revoice_jobs.json logs/zh_revoice_v1
log "REVOICE_DONE"
