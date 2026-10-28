#!/usr/bin/env bash
# Paper v2 experiments (make_v2_jobs.py): cascade baselines (decode, score, speech) and voice similarity of every
# system. Waits for run_speech_fix.sh (S-T speech must exist for its voice job), stops the holders on all cards, runs
# the gpuq6 queue, restores the holders. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SCR=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/d490b1cc-23c2-464f-a450-9958e1eda107/scratchpad
OLD=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
for d in C-van C-ft; do [ -e eval_outputs/uniss_qwen3_omni_v1/gstar_cascade_20261028/$d ] && { log "拒绝覆盖已有输出: $d"; exit 1; }; done
until grep -q -E "FIX_DONE|failed" logs/gstar_speech_fix_run.log; do sleep 30; done
log "holders off"
bash ${OLD}/holder_off.sh all
sleep 20
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u ${SCR}/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/v2_jobs.json logs/gstar_v2
log "QUEUE_DONE"
sleep 15
bash ${OLD}/holder_on.sh 0,1,3,4
bash ${OLD}/holder_on.sh 2,5,6,7
log "V2_DONE"
