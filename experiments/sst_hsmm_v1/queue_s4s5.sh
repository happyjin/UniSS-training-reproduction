#!/usr/bin/env bash
# overnight 2026-11-26/27: after the S3 chain (pid $1) exits -> S4 (change 50, ff_pipe2 TAG d3cma) -> S5 (change 51, p3d3s05q_report.sh).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "queue $$: waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
log "S3 chain ended; S4 starts"
TAG=d3cma DL=$D/d3/peft/a0.5000 PRM=$D/d3s05/h/params.json XFLAGS="--cma --lora-final" bash experiments/sst_hsmm_v1/ff_pipe2.sh > logs/sst_hsmm_v1/d3cma.log 2>&1 < /dev/null
log "S4 ended rc=$?; S5 starts"
bash experiments/sst_hsmm_v1/p3d3s05q_report.sh > logs/sst_hsmm_v1/p3d3s05q_report.log 2>&1 < /dev/null
log "S5 ended rc=$?"
touch $D/QUEUE_S4S5_DONE
