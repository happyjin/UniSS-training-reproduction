#!/usr/bin/env bash
# overnight 2026-11-27: S6 (change 52, p3hkm_report.sh) after the S3-remainder queue (pid $1) exits.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "queue $$: waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
bash experiments/sst_hsmm_v1/p3hkm_report.sh > logs/sst_hsmm_v1/p3hkm_report.log 2>&1 < /dev/null
log "S6 ended rc=$?"
touch eval_outputs/sst_hsmm_v1/QUEUE_S6_DONE
