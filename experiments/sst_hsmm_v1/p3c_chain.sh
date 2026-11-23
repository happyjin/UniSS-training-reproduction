#!/usr/bin/env bash
# sst_hsmm_v1 P3'' chain (prereg change 7): waits for P2'' (p2c/DONE or FAILED; stops on p4/FAILED); GO -> starts
# p3c_orch.sh detached; NO-GO -> log and stop. Starts nothing else. Checked with bash -n / DRY_RUN only.
#   setsid nohup bash experiments/sst_hsmm_v1/p3c_chain.sh > logs/sst_hsmm_v1/p3c_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
P4=eval_outputs/sst_hsmm_v1/p4; P2=eval_outputs/sst_hsmm_v1/p2c
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then log "[dry] wait p2c -> GO? -> p3c_orch.sh"; exit 0; fi
log "chain $$: waiting for P2''"
until [ -e $P2/DONE ] || [ -e $P2/FAILED ]; do [ -e $P4/FAILED ] && { log "P4 FAILED: stop"; exit 1; }; sleep 60; done
[ -e $P2/FAILED ] && { log "P2'' FAILED: stop"; exit 1; }
go=$(python3 -c "import json; print(int(json.load(open('$P2/summary.json'))['verdict']['GO']))" 2>/dev/null)
[ "$go" = 1 ] || { log "P2'' NO-GO (go=$go): stop, no P3''"; exit 0; }
setsid nohup bash experiments/sst_hsmm_v1/p3c_orch.sh > logs/sst_hsmm_v1/p3c_orch.log 2>&1 < /dev/null 8>&- 9>&- &
log "P2'' GO: p3c_orch pid $!"
