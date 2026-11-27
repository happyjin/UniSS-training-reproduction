#!/usr/bin/env bash
# overnight 2026-11-27: S3 (change 49) after the S4/S5 queue (pid $1) exits: rollouts + HSMM fit on p2f/merged (drafter_pipe2,
# TAG p2fx; the first attempt stopped because TAG p2f collided with the 11-23 P2f directories) -> ff_pipe TAG p2fxff (test p3p2fxff).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "queue $$: waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
env -u WAIT TAG=p2fx DTH=$D/p2f/merged STOP_AFTER_FIT=1 bash experiments/sst_hsmm_v1/drafter_pipe2.sh > logs/sst_hsmm_v1/p2fx.log 2>&1 < /dev/null
[ -e $D/p2fx/h/FIT_DONE ] || { log "fit did not finish"; exit 1; }
bash experiments/sst_hsmm_v1/make_ff_report.sh p2fxff $D/p2f/peft/a1.0000 $D/p2fx/h/params.json || { log "make report"; exit 1; }
env -u WAIT TAG=p2fxff DL=$D/p2f/peft/a1.0000 PRM=$D/p2fx/h/params.json bash experiments/sst_hsmm_v1/ff_pipe.sh > logs/sst_hsmm_v1/p2fxff.log 2>&1 < /dev/null
log "S3 ended rc=$?"
touch $D/QUEUE_S3REST_DONE
