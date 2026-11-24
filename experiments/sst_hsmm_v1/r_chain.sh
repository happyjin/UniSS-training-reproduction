#!/usr/bin/env bash
# prereg change 22 (CR-LA) chain: r_orch -> dev p2r -> test p3r (combo vs N, report only).
#   setsid nohup bash experiments/sst_hsmm_v1/r_chain.sh > logs/sst_hsmm_v1/r_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/r_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] r_orch -> p2r -> p3r"; exit 0; }
[ -e $L/r_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: train"
bash experiments/sst_hsmm_v1/r_orch.sh > $L/r_orch.log 2>&1 < /dev/null; [ -e $D/r/DONE ] || fail "r train"
log "dev"
bash experiments/sst_hsmm_v1/p2r_orch.sh > $L/p2r_orch.log 2>&1 < /dev/null; [ -e $D/p2r/DONE ] || fail "p2r"
log "test"
bash experiments/sst_hsmm_v1/p3r_report.sh > $L/p3r_report.log 2>&1 < /dev/null; [ -e $D/p3r/DONE ] || fail "p3r"
log "chain done"
