#!/usr/bin/env bash
# Paper v3 evaluation (make_v3_jobs.py, gpuq6 on all eight cards) once run_bidir_pipeline.sh has trained O2, S2 and O2
# without speaker stream; the GPU holders come back at the end. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
[ -e eval_outputs/uniss_qwen3_omni_v1/gstar_v3_20261029 ] && { log "拒绝覆盖已有输出: gstar_v3_20261029"; exit 1; }
until grep -q -E "BIDIR_PIPELINE_DONE|FAILED|failed|A8_FAIL" logs/bidir_pipeline_v8b.log; do sleep 60; done
grep -q BIDIR_PIPELINE_DONE logs/bidir_pipeline_v8b.log || { log "training chain did not finish"; exit 1; }
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/v3_jobs.json logs/gstar_v3_eval
log "QUEUE_DONE"
sleep 15
bash scripts/gpu_queue/holder_on.sh 0,1,3,4
bash scripts/gpu_queue/holder_on.sh 2,5,6,7
log "V3_EVAL_DONE"
