#!/usr/bin/env bash
# overnight 2026-11-27: S8 (change 54, s8_pipe.sh TAG aac) after the S7 queue (pid $1) exits.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "queue $$: waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
env -u WAIT TAG=aac DL=eval_outputs/sst_hsmm_v1/p2f/peft/a1.0000 bash experiments/sst_hsmm_v1/s8_pipe.sh > logs/sst_hsmm_v1/aac.log 2>&1 < /dev/null
log "S8 ended rc=$?"
touch eval_outputs/sst_hsmm_v1/QUEUE_S8_DONE
